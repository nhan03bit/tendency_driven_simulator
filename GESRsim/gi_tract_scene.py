"""
GI Tract scene builder — creates a simplified tubular GI-tract environment
in PyBullet for endoscope navigation training / testing.

The tract is composed of connected ring segments that form a curved tunnel.
Polyps (small spheres) can be placed on the inner wall for detection tasks.
"""

import math
import numpy as np


class GITractScene:
    """
    Build a simplified gastrointestinal tract inside an existing PyBullet
    simulation.

    Parameters
    ----------
    bullet : pybullet module
    inner_radius : float
        Inner radius of the tract tube (metres).  Default 15 mm.
    wall_thickness : float
        Thickness of each ring shell.  Default 2 mm.
    ring_spacing : float
        Distance between consecutive ring centres.  Default 8 mm.
    wall_color : list
        RGBA colour representing mucosal tissue.
    polyp_color : list
        RGBA for polyp spheres.
    """

    def __init__(
        self,
        bullet,
        inner_radius: float = 0.015,
        wall_thickness: float = 0.002,
        ring_spacing: float = 0.008,
        wall_color=None,
        polyp_color=None,
    ):
        self.bullet = bullet
        self.inner_radius = inner_radius
        self.wall_thickness = wall_thickness
        self.ring_spacing = ring_spacing
        self.wall_color = wall_color or [0.85, 0.55, 0.55, 0.45]
        self.polyp_color = polyp_color or [0.9, 0.2, 0.2, 1.0]

        self._ring_bodies = []
        self._polyp_bodies = []
        self._centreline = []  # list of (x, y, z) centres

    # ------------------------------------------------------------------
    # Tract geometry generators
    # ------------------------------------------------------------------
    def build_straight(self, start, direction, length):
        """
        Add a straight tube segment.

        Parameters
        ----------
        start : (3,) — start position
        direction : (3,) — unit direction vector
        length : float — total length in metres
        """
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        n_rings = max(int(length / self.ring_spacing), 2)
        for i in range(n_rings):
            centre = np.asarray(start) + d * i * self.ring_spacing
            self._add_ring(centre, d)
        return np.asarray(start) + d * (n_rings - 1) * self.ring_spacing

    def build_curve(self, start, start_dir, bend_angle, bend_radius, bend_axis=None):
        """
        Add a curved tube section (constant-curvature arc).

        Parameters
        ----------
        start : (3,) — start position
        start_dir : (3,) — tangent direction at entry
        bend_angle : float — total bend in radians (positive = left turn)
        bend_radius : float — radius of curvature in metres
        bend_axis : (3,) optional — axis about which to bend.
                    If None, defaults to the cross product of start_dir × [0,0,1].

        Returns end position and end tangent for chaining.
        """
        d = np.asarray(start_dir, dtype=float)
        d /= np.linalg.norm(d)
        s = np.asarray(start, dtype=float)

        if bend_axis is None:
            up = np.array([0.0, 0.0, 1.0])
            bend_axis = np.cross(d, up)
            if np.linalg.norm(bend_axis) < 1e-8:
                bend_axis = np.array([0.0, 1.0, 0.0])
        bend_axis = np.asarray(bend_axis, dtype=float)
        bend_axis /= np.linalg.norm(bend_axis)

        arc_length = abs(bend_angle) * bend_radius
        n_rings = max(int(arc_length / self.ring_spacing), 2)
        angles = np.linspace(0, bend_angle, n_rings)

        # Centre of curvature
        normal = np.cross(bend_axis, d)
        normal /= np.linalg.norm(normal)
        centre_of_curvature = s + normal * bend_radius

        end_pos = s
        end_dir = d
        for ang in angles:
            # Rotate d and offset around bend_axis
            R = self._rotation_matrix(bend_axis, ang)
            cur_dir = R @ d
            cur_pos = centre_of_curvature - R @ (normal * bend_radius)
            self._add_ring(cur_pos, cur_dir)
            end_pos = cur_pos
            end_dir = cur_dir

        return end_pos, end_dir

    def build_s_curve(self, start, start_dir, angle1, radius1, angle2, radius2,
                      bend_axis=None):
        """Convenience: two opposite bends forming an S-shape."""
        p1, d1 = self.build_curve(start, start_dir, angle1, radius1, bend_axis)
        p2, d2 = self.build_curve(p1, d1, -angle2, radius2, bend_axis)
        return p2, d2

    # ------------------------------------------------------------------
    # Polyps
    # ------------------------------------------------------------------
    def add_polyp(self, position, radius=0.003):
        """Place a small sphere polyp at the given position."""
        col = self.bullet.createCollisionShape(self.bullet.GEOM_SPHERE, radius=radius)
        vis = self.bullet.createVisualShape(
            self.bullet.GEOM_SPHERE, radius=radius, rgbaColor=self.polyp_color,
        )
        body = self.bullet.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis, basePosition=position,
        )
        self._polyp_bodies.append(body)
        return body

    def add_polyps_along_wall(self, n=5, rng=None):
        """
        Place *n* polyps randomly on the inner wall of existing rings.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        if len(self._centreline) < 2:
            return

        for _ in range(n):
            idx = rng.integers(0, len(self._centreline))
            c = self._centreline[idx]
            # Random angle around the ring
            theta = rng.uniform(0, 2 * np.pi)
            # Offset on the inner wall (perpendicular to tract axis — approx)
            offset = self.inner_radius * 0.85 * np.array([
                math.cos(theta), math.sin(theta), 0,
            ])
            pos = np.array(c) + offset
            self.add_polyp(pos)

    # ------------------------------------------------------------------
    # Pre-built anatomical layouts
    # ------------------------------------------------------------------
    def build_colon_layout(self, origin=None):
        """
        Build a simplified colon-like tract:
        straight ascending → hepatic flexure → transverse →
        splenic flexure → descending straight.
        """
        origin = np.asarray(origin) if origin is not None else np.array([0.0, 0.0, 0.02])
        d = np.array([0.0, 0.0, 1.0])  # ascending

        # Ascending colon
        p = self.build_straight(origin, d, length=0.08)

        # Hepatic flexure (90° bend)
        p, d = self.build_curve(p, d, bend_angle=np.pi / 2, bend_radius=0.03,
                                bend_axis=np.array([0, 1, 0]))

        # Transverse colon
        p = self.build_straight(p, d, length=0.10)

        # Splenic flexure (90° bend)
        p, d = self.build_curve(p, d, bend_angle=np.pi / 2, bend_radius=0.03,
                                bend_axis=np.array([0, 1, 0]))

        # Descending colon
        p = self.build_straight(p, d, length=0.08)

        # Add some polyps
        self.add_polyps_along_wall(n=6)

        return p, d

    def build_simple_tunnel(self, origin=None):
        """
        A simpler S-shaped tunnel for quick testing.
        """
        origin = np.asarray(origin) if origin is not None else np.array([0.0, 0.0, 0.02])
        d = np.array([0.0, 0.0, 1.0])

        p = self.build_straight(origin, d, length=0.06)
        p, d = self.build_curve(p, d, bend_angle=np.pi / 3, bend_radius=0.04,
                                bend_axis=np.array([0, 1, 0]))
        p = self.build_straight(p, d, length=0.05)
        p, d = self.build_curve(p, d, bend_angle=-np.pi / 4, bend_radius=0.05,
                                bend_axis=np.array([0, 1, 0]))
        p = self.build_straight(p, d, length=0.06)

        self.add_polyps_along_wall(n=4)
        return p, d

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_ring(self, centre, direction):
        """
        Add a single torus-like ring at *centre* oriented along *direction*.
        Approximated as a set of small boxes arranged in a circle.
        """
        centre = np.asarray(centre, dtype=float)
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        self._centreline.append(tuple(centre))

        n_boxes = 12  # boxes per ring
        r = self.inner_radius
        wt = self.wall_thickness
        box_arc = 2 * np.pi / n_boxes
        box_width = r * box_arc  # tangential width

        # Orthogonal frame
        d = direction
        if abs(d[2]) < 0.9:
            perp1 = np.cross(d, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(d, np.array([1, 0, 0]))
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(d, perp1)

        col = self.bullet.createCollisionShape(
            self.bullet.GEOM_BOX,
            halfExtents=[box_width / 2, wt / 2, self.ring_spacing / 2],
        )
        vis = self.bullet.createVisualShape(
            self.bullet.GEOM_BOX,
            halfExtents=[box_width / 2, wt / 2, self.ring_spacing / 2],
            rgbaColor=self.wall_color,
        )

        for k in range(n_boxes):
            theta = k * box_arc
            offset = r * (math.cos(theta) * perp1 + math.sin(theta) * perp2)
            pos = centre + offset

            # Orient box so its local-z aligns with tract direction
            # and local-y points radially outward
            radial = offset / np.linalg.norm(offset)
            tangent = np.cross(d, radial)
            if np.linalg.norm(tangent) < 1e-8:
                tangent = perp1
            tangent /= np.linalg.norm(tangent)

            # Build rotation matrix columns: x=tangent, y=radial, z=direction
            rot = np.column_stack([tangent, radial, d])
            # Ensure proper rotation (det=+1)
            if np.linalg.det(rot) < 0:
                rot[:, 0] *= -1

            quat = self._rotmat_to_quat(rot)
            body = self.bullet.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos, baseOrientation=quat,
            )
            self._ring_bodies.append(body)

    @staticmethod
    def _rotation_matrix(axis, angle):
        """Rodrigues rotation matrix around *axis* by *angle* radians."""
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

    @staticmethod
    def _rotmat_to_quat(R):
        """Convert 3×3 rotation matrix to [x, y, z, w] quaternion."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 2.0 * math.sqrt(tr + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return [x, y, z, w]
