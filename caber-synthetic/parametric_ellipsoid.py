# Original Implementation from John Weis, see
# https://github.com/johnweis0480/Parametric_Pytorch
import numpy as np
import torch
from torch.autograd.functional import jacobian, jvp
import scipy.sparse as sp
from simpeg.maps import BaseParametric

class PytorchMapping(BaseParametric):
    """
    Original Implementation from John Weis, see
    https://github.com/johnweis0480/Parametric_Pytorch
    """

    def __init__(self, mesh=None, nP=None, active_cells= None, forward_transform=None, inverse_transform=None,params = None,**kwargs):
        self.mesh = mesh

        self._nP = nP
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.active_cells = active_cells
        self.params = params
        self.xyz = np.vstack((self.x,self.y,self.z)).T

    def _transform(self, m):
        '''
        model parameters (parametric or latent dimensionality)
        '''
        m_loc = torch.Tensor(m)

        if self.params is not None:
            return self.forward_transform(m_loc,self.params,self.xyz).numpy()
        else:
            return self.forward_transform(m_loc).numpy()



    def deriv(self, m, v=None):
        m_loc = torch.Tensor(m)

        if v is not None:
            v_loc = torch.Tensor(v)
            return sp.csr_matrix(jvp(lambda m_loc: self.forward_transform(m_loc,self.params,self.xyz),m_loc,v_loc)[1].numpy())
        else:
            return sp.csr_matrix(jacobian(lambda m_loc: self.forward_transform(m_loc,self.params,self.xyz),m_loc,strategy='forward-mode',vectorize=True).numpy())

    @property
    def nP(self):
        return self._nP

    @property
    def shape(self):
        if self.active_cells is not None:
            return (self.active_cells.sum(), self._nP)
        else:
            return (self.mesh.n_cells, self._nP)

#Parametric Ellipse function, could be any parameterization, needs to be written in pytorch
def ellipsoid_torch_transform(m, params, xyz):
    xyz = xyz
    c = params[0]
    n_ellipse = params[1]
    p_0 = m[0]

    X = torch.tensor(xyz[:, 0])
    Y = torch.tensor(xyz[:, 1])
    Z = torch.tensor(xyz[:, 2])

    xyz = torch.vstack((X, Y, Z))

    # Collect a "membership logit" and a property value for the background and
    # for each ellipsoid, then combine them with a softmax over the *logits*
    # (how far inside each ellipsoid a point is) rather than over the values.
    # This selects an ellipsoid wherever it is active, regardless of whether
    # its value is above OR below the background -- unlike a softmax over the
    # values, which can only ever favor the larger value.
    n_pts = xyz.shape[1]
    logits = [torch.zeros(n_pts, dtype=torch.float64)]  # background: logit 0
    prop_values = [p_0]                                 # background value

    for i in range(n_ellipse):
        rx, ry, rz, phix, phiy, phiz, x_0, y_0, z_0, p_1, a = m[1 + i * 11:1 + (i + 1) * 11]

        xyz_0 = torch.vstack((x_0, y_0, z_0))

        S = torch.zeros((3, 3), dtype=torch.float64)
        S[0, 0] = 2 / rx
        S[1, 1] = 2 / ry
        S[2, 2] = 2 / rz

        Rx = torch.zeros_like(S)
        Rx[0, 0] = 1
        Rx[1, 1] = torch.cos(phix)
        Rx[1, 2] = -torch.sin(phix)
        Rx[2, 2] = torch.cos(phix)
        Rx[2, 1] = torch.sin(phix)

        Ry = torch.zeros_like(S)
        Ry[1, 1] = 1
        Ry[0, 0] = torch.cos(phiy)
        Ry[2, 0] = -torch.sin(phiy)
        Ry[2, 2] = torch.cos(phiy)
        Ry[0, 2] = torch.sin(phiy)

        Rz = torch.zeros_like(S)
        Rz[2, 2] = 1
        Rz[0, 0] = torch.cos(phiz)
        Rz[0, 1] = -torch.sin(phiz)
        Rz[1, 1] = torch.cos(phiz)
        Rz[1, 0] = torch.sin(phiz)

        T = S @ Rx @ Ry @ Rz
        M = T.T @ T

        xyz_m_xyz_0 = xyz - xyz_0
        tau = M @ (xyz_m_xyz_0)

        tau = c - torch.sum(xyz_m_xyz_0 * tau, dim=0)

        # tau > 0 inside the ellipsoid, so this logit is large there. The
        # factor of 2 reproduces the boundary half-width of the original
        # 0.5 * (1 + tanh(a * tau)) transition.
        logits.append(2.0 * a * tau)
        prop_values.append(p_1)

    logits = torch.stack(logits)            # (n_ellipse + 1, n_pts)
    prop_values = torch.stack(prop_values)  # (n_ellipse + 1,)
    # Softmax over the logits is a smooth, differentiable partition of unity
    # that picks the most-inside ellipsoid (or the background) at each point.
    weights = torch.softmax(logits, dim=0)  # numerically stable
    p = torch.sum(weights * prop_values[:, None], dim=0)
    return p

class ParametricEllipsoid(PytorchMapping):
    def __init__(
            self, mesh=None, active_cells= None, smoothness_factor=1.,
    ):
        super().__init__(
            mesh=mesh,
            nP=12,
            active_cells=active_cells,
            forward_transform=ellipsoid_torch_transform,
            inverse_transform=None,
            params=[smoothness_factor, 1]  # c and n_ellipse
        )
