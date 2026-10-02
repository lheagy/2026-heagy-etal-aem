# Original implementation by John Weis:
# https://github.com/johnweis0480/Parametric_Pytorch
#
# Refactored for inversion use:
#   * dense Jacobians (no csr wrapping of dense results),
#   * xyz cached as a torch tensor once,
#   * float64 throughout,
#   * log semi-axes (positivity + better-scaled Jacobian columns),
#   * boundary sharpness is a fixed hyperparameter (no longer in `m`),
#     so the boundary-sharpness <-> interior-value degeneracy is removed,
#   * the redundant `c` level-set parameter is dropped; the boundary is
#     fixed at the level set <x-x0, M (x-x0)> = 1, so r_x, r_y, r_z are
#     the actual semi-axes (NOT full axes as in the original).
#
# Parameter vector layout (length 1 + 10 * n_ellipsoids):
#     m[0]                       = background property
#     m[1 + 10*i : 11 + 10*i]    = per-ellipsoid block i, with entries
#         (log_rx, log_ry, log_rz,
#          phi_x, phi_y, phi_z,
#          x_0, y_0, z_0,
#          p_interior)
import numpy as np
import torch
from torch.autograd.functional import jacobian, jvp
from simpeg.maps import BaseParametric


class PytorchMapping(BaseParametric):
    """Differentiable parametric mapping with a PyTorch forward transform.

    Subclasses implement ``forward_transform(self, m_t)``: a function that
    takes a 1-D torch tensor of model parameters and returns a 1-D torch
    tensor of property values on the active cells. Derivatives are
    obtained via forward-mode autodiff, which is the right choice when
    ``nP`` is small and the output is large.
    """

    def __init__(self, mesh, nP, active_cells=None, **kwargs):
        super().__init__(mesh=mesh, active_cells=active_cells, **kwargs)
        self._nP = int(nP)
        # cache active cell centers as a (3, n_pts) float64 torch tensor
        xyz = np.vstack((self.x, self.y, self.z))
        self._xyz = torch.as_tensor(xyz, dtype=torch.float64)

    @property
    def nP(self):
        return self._nP

    @property
    def shape(self):
        return (len(self.x), self._nP)

    def forward_transform(self, m_t):
        raise NotImplementedError

    def _transform(self, m):
        m_t = torch.as_tensor(m, dtype=torch.float64)
        with torch.no_grad():
            return self.forward_transform(m_t).numpy()

    def deriv(self, m, v=None):
        m_t = torch.as_tensor(m, dtype=torch.float64)
        if v is not None:
            v_t = torch.as_tensor(v, dtype=torch.float64)
            return jvp(self.forward_transform, m_t, v_t)[1].numpy()
        return jacobian(
            self.forward_transform, m_t,
            strategy="forward-mode", vectorize=True,
        ).numpy()


def _rotation_matrix(phi_x, phi_y, phi_z):
    """R = Rx @ Ry @ Rz with intrinsic Euler angles (radians)."""
    one = torch.ones_like(phi_x)
    zero = torch.zeros_like(phi_x)
    cx, sx = torch.cos(phi_x), torch.sin(phi_x)
    cy, sy = torch.cos(phi_y), torch.sin(phi_y)
    cz, sz = torch.cos(phi_z), torch.sin(phi_z)
    Rx = torch.stack([
        torch.stack([one,  zero, zero]),
        torch.stack([zero,   cx,  -sx]),
        torch.stack([zero,   sx,   cx]),
    ])
    Ry = torch.stack([
        torch.stack([  cy, zero,   sy]),
        torch.stack([zero,  one, zero]),
        torch.stack([ -sy, zero,   cy]),
    ])
    Rz = torch.stack([
        torch.stack([  cz,  -sz, zero]),
        torch.stack([  sz,   cz, zero]),
        torch.stack([zero, zero,  one]),
    ])
    return Rx @ Ry @ Rz


class ParametricEllipsoid(PytorchMapping):
    """Parametric ellipsoid(s) embedded in a homogeneous background.

    The boundary is the level set
        (x - x_0)^T M (x - x_0) = 1
    with
        M = R^T diag(1/r_x^2, 1/r_y^2, 1/r_z^2) R,
        R = Rx @ Ry @ Rz.
    Inside/outside is smoothed with a softmax over per-ellipsoid logits,
    giving a partition of unity that handles interior values either above
    or below the background.

    Boundary sharpness is controlled by ``boundary_sharpness`` and is
    *fixed* (not inverted for) -- inverting for it together with the
    interior property created a degeneracy where a diffuse boundary +
    extreme interior value mimics a sharp boundary + correct value.

    Semi-axis convention: ``r_x, r_y, r_z`` are the **semi-axes** (the
    distance from center to the ellipsoid surface along each principal
    axis). The original implementation used "full axes" (2 * semi-axis);
    callers porting from the old layout should halve their starting
    semi-axis magnitudes accordingly, or equivalently use
    ``log(old_r / 2)`` instead of ``log(old_r)``.
    """

    BLOCK_SIZE = 10
    BLOCK_NAMES = (
        "log_rx", "log_ry", "log_rz",
        "phi_x", "phi_y", "phi_z",
        "x_0", "y_0", "z_0",
        "p_interior",
    )

    def __init__(
        self,
        mesh,
        active_cells=None,
        n_ellipsoids=1,
        boundary_sharpness=1.0,
    ):
        self.n_ellipsoids = int(n_ellipsoids)
        self.boundary_sharpness = float(boundary_sharpness)
        super().__init__(
            mesh=mesh,
            nP=1 + self.BLOCK_SIZE * self.n_ellipsoids,
            active_cells=active_cells,
        )

    def _block(self, m_t, i):
        start = 1 + i * self.BLOCK_SIZE
        return m_t[start : start + self.BLOCK_SIZE]

    def forward_transform(self, m_t):
        p_bg = m_t[0]
        n_pts = self._xyz.shape[1]

        logits = [torch.zeros(n_pts, dtype=torch.float64)]
        props = [p_bg]

        for i in range(self.n_ellipsoids):
            b = self._block(m_t, i)
            log_rx, log_ry, log_rz = b[0], b[1], b[2]
            phi_x,  phi_y,  phi_z  = b[3], b[4], b[5]
            x_0,    y_0,    z_0    = b[6], b[7], b[8]
            p_in                   = b[9]

            inv_r = torch.stack([
                torch.exp(-log_rx),
                torch.exp(-log_ry),
                torch.exp(-log_rz),
            ])
            S = torch.diag(inv_r)
            R = _rotation_matrix(phi_x, phi_y, phi_z)
            T = S @ R
            M = T.T @ T

            center = torch.stack([x_0, y_0, z_0]).unsqueeze(1)
            dx = self._xyz - center
            # tau > 0 inside the ellipsoid, boundary at tau = 0
            tau = 1.0 - torch.sum(dx * (M @ dx), dim=0)

            logits.append(self.boundary_sharpness * tau)
            props.append(p_in)

        logits = torch.stack(logits)            # (n_ellipsoids + 1, n_pts)
        props = torch.stack(props)              # (n_ellipsoids + 1,)
        weights = torch.softmax(logits, dim=0)  # numerically stable
        return torch.sum(weights * props[:, None], dim=0)
