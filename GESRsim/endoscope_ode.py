"""
EndoscopeODE — Cosserat-rod kinematics for a multi-segment tendon-driven
gastrointestinal endoscope, inspired by the GESRsim paper.

Each segment is actuated by **4 tendons** arranged at 90° intervals around the
backbone.  A pair of antagonistic tendons produces bending in one plane:

    * Tendons 1 & 3 → bending in the X–Z plane  (curvature ux)
    * Tendons 2 & 4 → bending in the Y–Z plane  (curvature uy)

The backbone length itself can change (insertion / retraction), so the state
per segment is  [dl, dt1, dt2, dt3, dt4]  where:
    dl   — change in backbone arc-length  (positive = extension)
    dt_i — change in tendon-i length      (positive = tendon pulled)

This class wraps the SoftManiSim ODE solver and adds endoscope-specific
geometry and a convenience method that converts 4-tendon displacements into
the (dl, ux, uy) representation the Cosserat rod solver expects.
"""

import numpy as np
from scipy.integrate import solve_ivp


class EndoscopeODE:
    """Cosserat-rod forward-kinematics for one or more endoscope segments."""

    def __init__(
        self,
        num_segments: int = 2,
        segment_length: float = 60e-3,      # 60 mm per segment
        tendon_offset: float = 4.0e-3,       # 4 mm radial offset
        outer_radius: float = 6.0e-3,        # 6 mm tube radius
        ds: float = 0.001,                   # integration step (m)
    ):
        self.num_segments = num_segments
        self.segment_length = segment_length  # l0 for each segment
        self.tendon_offset = tendon_offset    # d
        self.outer_radius = outer_radius
        self.ds = ds

        # Internal ODE state
        self._reset_y0()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _reset_y0(self):
        """Reset to straight configuration at the origin."""
        r0 = np.zeros((3, 1))
        R0 = np.eye(3).reshape(9, 1)
        y0 = np.concatenate((r0, R0), axis=0)
        self.y0 = np.squeeze(np.asarray(y0))

    # ------------------------------------------------------------------
    # Tendon → curvature conversion
    # ------------------------------------------------------------------
    def tendon_to_curvature(self, tendon_deltas: np.ndarray, seg_idx: int = 0):
        """
        Convert 4 tendon displacements to (l, ux, uy) for the Cosserat ODE.

        Parameters
        ----------
        tendon_deltas : (4,) array — [dt1, dt2, dt3, dt4]
            Pull (+) or release (−) of each tendon.
        seg_idx : int
            Segment index (reserved for segment-specific lengths).

        Returns
        -------
        l  : float  — total arc-length of the segment
        ux : float  — curvature about X
        uy : float  — curvature about Y
        """
        dt1, dt2, dt3, dt4 = tendon_deltas
        d = self.tendon_offset
        l = self.segment_length   # backbone length stays constant here

        # Differential tendon pairs → bending curvatures
        #   tendons 1,3 at ±d along local-y  →  bending about local-x  (uy)
        #   tendons 2,4 at ±d along local-x  →  bending about local-y  (ux)
        uy = (dt1 - dt3) / (l * d)
        ux = (dt2 - dt4) / -(l * d)
        return l, ux, uy

    # ------------------------------------------------------------------
    # Core ODE
    # ------------------------------------------------------------------
    def _ode_function(self, s, y, ux, uy):
        """Cosserat rod ODE  dy/ds = f(s, y)."""
        dydt = np.zeros(12)
        e3 = np.array([0, 0, 1]).reshape(3, 1)
        u_hat = np.array([
            [0,   0,   uy],
            [0,   0,  -ux],
            [-uy, ux,  0 ],
        ])
        R = np.array([y[3:6], y[6:9], y[9:12]]).reshape(3, 3)
        dR = R @ u_hat
        dr = R @ e3
        dRR = dR.T
        dydt[0:3]  = dr.T
        dydt[3:6]  = dRR[:, 0]
        dydt[6:9]  = dRR[:, 1]
        dydt[9:12] = dRR[:, 2]
        return dydt

    def _solve_segment(self, l, ux, uy):
        """Integrate one segment and return the full solution array (12, N)."""
        t_eval = np.linspace(0, l, max(int(l / self.ds), 2))
        sol = solve_ivp(
            lambda s, y: self._ode_function(s, y, ux, uy),
            (0, l),
            self.y0,
            t_eval=t_eval,
        )
        self.y0 = sol.y[:, -1]   # next segment starts here
        return sol.y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, actions: np.ndarray):
        """
        Compute the full backbone curve for all segments.

        Parameters
        ----------
        actions : (num_segments, 4) or (num_segments*4,) array
            Per-segment tendon displacements [dt1, dt2, dt3, dt4].

        Returns
        -------
        sol : (12, N) ndarray — concatenated backbone states
            rows 0-2 : position  (x, y, z) in metres
            rows 3-11: flattened rotation matrix columns
        segment_indices : list of int
            Column index where each new segment starts in *sol*.
        """
        actions = np.asarray(actions).reshape(self.num_segments, 4)
        self._reset_y0()

        sols = []
        seg_indices = [0]
        for i, td in enumerate(actions):
            l, ux, uy = self.tendon_to_curvature(td, seg_idx=i)
            seg_sol = self._solve_segment(l, ux, uy)
            sols.append(seg_sol)
            seg_indices.append(seg_indices[-1] + seg_sol.shape[1])

        full_sol = np.concatenate(sols, axis=1)
        return full_sol, seg_indices

    def forward_simple(self, actions: np.ndarray):
        """
        Convenience wrapper — accepts the same (dl, uy_pull, ux_pull)
        3-per-segment format that the base SoftManiSim ODE uses.

        Parameters
        ----------
        actions : (num_segments * 3,) array
            [dl_1, pull_y_1, pull_x_1,  dl_2, pull_y_2, pull_x_2, ...]

        Returns
        -------
        sol : (12, N) ndarray
        """
        actions = np.asarray(actions).reshape(self.num_segments, 3)
        self._reset_y0()
        sols = []
        for act in actions:
            dl, pull_y, pull_x = act
            l = self.segment_length + dl
            d = self.tendon_offset
            uy = pull_y / (l * d)
            ux = pull_x / -(l * d)
            seg_sol = self._solve_segment(l, ux, uy)
            sols.append(seg_sol)
        return np.concatenate(sols, axis=1)

    def tip_position(self, actions):
        """Return just the tip (x,y,z) in segment-base frame."""
        sol, _ = self.forward(actions)
        return sol[0:3, -1]

    def tip_jacobian(self, actions, eps=1e-5):
        """
        Numerical Jacobian  J = d(tip_pos) / d(actions)   (3 × 4*N_seg).
        """
        actions = np.asarray(actions).ravel()
        n = len(actions)
        p0 = self.tip_position(actions.reshape(-1, 4))
        J = np.zeros((3, n))
        for i in range(n):
            da = np.zeros(n)
            da[i] = eps
            p_plus  = self.tip_position((actions + da).reshape(-1, 4))
            p_minus = self.tip_position((actions - da).reshape(-1, 4))
            J[:, i] = (p_plus - p_minus) / (2 * eps)
        return J
