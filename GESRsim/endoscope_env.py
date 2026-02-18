"""
EndoscopeEnv — PyBullet-based simulation environment for a tendon-driven
gastrointestinal endoscope, inspired by the GESRsim paper.

The endoscope is rendered as a chain of **cylinders** (not spheres) to mimic
the smooth, tubular look of a real flexible endoscope.  An eye-in-hand camera
is placed at the tip to simulate the endoscope camera view.

The class follows the same API patterns as SoftManiSim's
``SoftRobotBasicEnvironment`` so it integrates seamlessly with the rest of
the framework.
"""

import math
import numpy as np
import pybullet_data
import cv2

from GESRsim.endoscope_ode import EndoscopeODE
from pybullet_env.camera.camera import Camera


class EndoscopeEnv:
    """
    Tendon-driven endoscope simulation in PyBullet.

    Parameters
    ----------
    bullet : pybullet module or None
        If None a new PyBullet GUI session is created.
    num_segments : int
        Number of independently actuated bending segments (default 2).
    segment_length : float
        Rest length of each segment in metres (default 60 mm).
    tendon_offset : float
        Radial distance from backbone to each tendon (default 4 mm).
    outer_radius : float
        Visual / collision radius of the endoscope tube (default 5 mm).
    num_cylinders : int
        Number of short cylinder links used to render the tube (default 40).
    body_color : list
        RGBA colour for the tube body.
    tip_color : list
        RGBA colour for the distal tip.
    camera_enabled : bool
        Whether to create the tip-mounted endoscope camera.
    """

    def __init__(
        self,
        bullet=None,
        num_segments: int = 2,
        segment_length: float = 60e-3,
        tendon_offset: float = 4.0e-3,
        outer_radius: float = 5.0e-3,
        num_cylinders: int = 40,
        body_color=None,
        tip_color=None,
        camera_enabled: bool = True,
        gui: bool = True,
    ):
        self._sim_dt = 0.005
        self._gui = gui
        self._num_segments = num_segments
        self._outer_radius = outer_radius
        self._num_cylinders = num_cylinders
        self._body_color = body_color or [0.75, 0.75, 0.78, 1.0]
        self._tip_color = tip_color or [0.15, 0.15, 0.15, 1.0]
        self._camera_enabled = camera_enabled

        # ----- PyBullet init -----
        if bullet is None:
            import pybullet as p
            self.bullet = p
            self.bullet.connect(self.bullet.GUI if gui else self.bullet.DIRECT)
            self.bullet.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.bullet.setGravity(0, 0, -9.81)
            self.bullet.setTimeStep(self._sim_dt)
            self.plane_id = self.bullet.loadURDF("plane.urdf")
            self.bullet.configureDebugVisualizer(self.bullet.COV_ENABLE_GUI, 0)
            self.bullet.resetDebugVisualizerCamera(
                cameraDistance=0.25, cameraYaw=45, cameraPitch=-25,
                cameraTargetPosition=[0.0, 0.15, 0.12],
            )
        else:
            self.bullet = bullet

        # ----- Endoscope kinematics -----
        self._ode = EndoscopeODE(
            num_segments=num_segments,
            segment_length=segment_length,
            tendon_offset=tendon_offset,
            outer_radius=outer_radius,
        )

        self._base_pos = np.array([0.0, 0.0, 0.0])
        self._base_ori = [0, 0, 0, 1]
        self._marker_id = None

        # Build visual bodies
        self._create_tube()

        # Tip camera
        if self._camera_enabled:
            self._tip_camera = None
            self._init_tip_camera(np.zeros(3), np.array([0, 0, 0.05]))

    # ==================================================================
    # Visual body creation
    # ==================================================================
    def _create_tube(self):
        """Create cylinder chain for the endoscope body + a sphere tip."""
        # Solve straight config
        zero_action = np.zeros(self._num_segments * 4)
        sol, _ = self._ode.forward(zero_action.reshape(-1, 4))

        # Compute per-cylinder length
        total_arc = self._num_segments * self._ode.segment_length
        cyl_half_len = (total_arc / self._num_cylinders) / 2.0
        r = self._outer_radius

        # Collision & visual shapes
        body_col = self.bullet.createCollisionShape(
            self.bullet.GEOM_CYLINDER, radius=r, height=2 * cyl_half_len,
        )
        body_vis = self.bullet.createVisualShape(
            self.bullet.GEOM_CYLINDER, radius=r, length=2 * cyl_half_len,
            rgbaColor=self._body_color,
        )
        tip_col = self.bullet.createCollisionShape(
            self.bullet.GEOM_SPHERE, radius=r * 1.15,
        )
        tip_vis = self.bullet.createVisualShape(
            self.bullet.GEOM_SPHERE, radius=r * 1.15,
            rgbaColor=self._tip_color,
        )

        # Sample positions along the backbone
        idx = np.linspace(0, sol.shape[1] - 1, self._num_cylinders, dtype=int)
        positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]  # swap y/z for pybullet

        self._tube_bodies = []
        for i, pos in enumerate(positions):
            if i < len(positions) - 1:
                ori = self._cylinder_orientation(positions[i], positions[i + 1])
            else:
                ori = self._cylinder_orientation(positions[-2], positions[-1])
            body = self.bullet.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=body_col,
                baseVisualShapeIndex=body_vis,
                basePosition=np.array(pos) + self._base_pos,
                baseOrientation=ori,
            )
            self._tube_bodies.append(body)

        # Tip sphere
        tip_body = self.bullet.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=tip_col,
            baseVisualShapeIndex=tip_vis,
            basePosition=np.array(positions[-1]) + self._base_pos,
        )
        self._tube_bodies.append(tip_body)
        self._dummy_sim_step(1)

    # ==================================================================
    # Motion API
    # ==================================================================
    def move(
        self,
        tendon_actions: np.ndarray,
        base_pos: np.ndarray = None,
        base_ori_euler: np.ndarray = None,
    ):
        """
        Compute FK and update the visual bodies.

        Parameters
        ----------
        tendon_actions : (num_segments, 4) or flat array
            Per-segment tendon displacements.
        base_pos : (3,) optional — world position of the endoscope base.
        base_ori_euler : (3,) optional — rpy of the base in radians.

        Returns
        -------
        tip_pos : (3,) — world-frame tip position
        tip_ori : (4,) — world-frame tip quaternion
        sol     : (12, N) — full ODE solution
        """
        if base_pos is not None:
            self._base_pos = np.asarray(base_pos)
        if base_ori_euler is not None:
            self._base_ori = self.bullet.getQuaternionFromEuler(base_ori_euler)

        tendon_actions = np.asarray(tendon_actions).reshape(self._num_segments, 4)
        sol, seg_idx = self._ode.forward(tendon_actions)

        idx = np.linspace(0, sol.shape[1] - 1, self._num_cylinders, dtype=int)
        positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]

        for i, pos in enumerate(positions):
            world_pos, world_ori = self.bullet.multiplyTransforms(
                self._base_pos, self._base_ori, pos, [0, 0, 0, 1],
            )
            if i < len(positions) - 1:
                next_pos = positions[i + 1]
                next_world, _ = self.bullet.multiplyTransforms(
                    self._base_pos, self._base_ori, next_pos, [0, 0, 0, 1],
                )
                cyl_ori = self._cylinder_orientation(world_pos, next_world)
            else:
                prev_pos = positions[-2]
                prev_world, _ = self.bullet.multiplyTransforms(
                    self._base_pos, self._base_ori, prev_pos, [0, 0, 0, 1],
                )
                cyl_ori = self._cylinder_orientation(prev_world, world_pos)

            self.bullet.resetBasePositionAndOrientation(
                self._tube_bodies[i], world_pos, cyl_ori,
            )

        # Tip sphere
        tip_local = positions[-1]
        tip_world, tip_ori_w = self.bullet.multiplyTransforms(
            self._base_pos, self._base_ori, tip_local, [0, 0, 0, 1],
        )
        self.bullet.resetBasePositionAndOrientation(
            self._tube_bodies[-1], tip_world, tip_ori_w,
        )

        # Update tip camera
        if self._camera_enabled:
            prev_world, _ = self.bullet.multiplyTransforms(
                self._base_pos, self._base_ori, positions[-2], [0, 0, 0, 1],
            )
            direction = np.array(tip_world) - np.array(prev_world)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            else:
                direction = np.array([0, 0, 1])
            cam_target = np.array(tip_world) + 0.05 * direction
            self._init_tip_camera(np.array(tip_world), cam_target)

        self.bullet.stepSimulation()

        self._tip_pos = np.array(tip_world)
        self._tip_ori = np.array(tip_ori_w)
        return self._tip_pos, self._tip_ori, sol

    def move_simple(
        self,
        actions: np.ndarray,
        base_pos: np.ndarray = None,
        base_ori_euler: np.ndarray = None,
    ):
        """
        Move using the 3-per-segment format: [dl, pull_y, pull_x] per segment.
        Internally converts to (12, N) backbone curve and updates visuals.
        """
        if base_pos is not None:
            self._base_pos = np.asarray(base_pos)
        if base_ori_euler is not None:
            self._base_ori = self.bullet.getQuaternionFromEuler(base_ori_euler)

        sol = self._ode.forward_simple(actions)

        idx = np.linspace(0, sol.shape[1] - 1, self._num_cylinders, dtype=int)
        positions = [(sol[0, i], sol[2, i], sol[1, i]) for i in idx]

        for i, pos in enumerate(positions):
            world_pos, world_ori = self.bullet.multiplyTransforms(
                self._base_pos, self._base_ori, pos, [0, 0, 0, 1],
            )
            if i < len(positions) - 1:
                next_pos = positions[i + 1]
                next_world, _ = self.bullet.multiplyTransforms(
                    self._base_pos, self._base_ori, next_pos, [0, 0, 0, 1],
                )
                cyl_ori = self._cylinder_orientation(world_pos, next_world)
            else:
                prev_pos = positions[-2]
                prev_world, _ = self.bullet.multiplyTransforms(
                    self._base_pos, self._base_ori, prev_pos, [0, 0, 0, 1],
                )
                cyl_ori = self._cylinder_orientation(prev_world, world_pos)

            self.bullet.resetBasePositionAndOrientation(
                self._tube_bodies[i], world_pos, cyl_ori,
            )

        # Tip sphere
        tip_local = positions[-1]
        tip_world, tip_ori_w = self.bullet.multiplyTransforms(
            self._base_pos, self._base_ori, tip_local, [0, 0, 0, 1],
        )
        self.bullet.resetBasePositionAndOrientation(
            self._tube_bodies[-1], tip_world, tip_ori_w,
        )

        # Update tip camera
        if self._camera_enabled:
            prev_world, _ = self.bullet.multiplyTransforms(
                self._base_pos, self._base_ori, positions[-2], [0, 0, 0, 1],
            )
            direction = np.array(tip_world) - np.array(prev_world)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            else:
                direction = np.array([0, 0, 1])
            cam_target = np.array(tip_world) + 0.05 * direction
            self._init_tip_camera(np.array(tip_world), cam_target)

        self.bullet.stepSimulation()
        self._tip_pos = np.array(tip_world)
        self._tip_ori = np.array(tip_ori_w)
        return self._tip_pos, self._tip_ori, sol

    # ==================================================================
    # Tip position (for Jacobian computation, no visual update)
    # ==================================================================
    def calc_tip_pos(self, tendon_actions):
        """Compute tip position without updating visuals (for Jacobian)."""
        tendon_actions = np.asarray(tendon_actions).reshape(self._num_segments, 4)
        sol, _ = self._ode.forward(tendon_actions)
        tip_local = (sol[0, -1], sol[2, -1], sol[1, -1])
        tip_world, _ = self.bullet.multiplyTransforms(
            self._base_pos, self._base_ori, tip_local, [0, 0, 0, 1],
        )
        return np.array(tip_world)

    # ==================================================================
    # Camera
    # ==================================================================
    def _init_tip_camera(self, cam_pos, cam_target):
        self._tip_camera = Camera(
            cam_pos=cam_pos, cam_target=cam_target,
            near=0.005, far=0.5, size=[320, 240], fov=120,
        )

    def capture_endoscope_image(self):
        """Get RGB + depth from the tip-mounted endoscope camera."""
        if self._tip_camera is None:
            return None, None
        bgr, depth, _ = self._tip_camera.get_cam_img()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, depth

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _cylinder_orientation(p1, p2):
        """Quaternion that aligns the Z-axis of a cylinder from p1 → p2."""
        diff = np.array(p2) - np.array(p1)
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            return [0, 0, 0, 1]
        d = diff / norm
        # Rotation that maps [0,0,1] → d
        # Using axis-angle: axis = cross([0,0,1], d), angle = acos(dot)
        z = np.array([0.0, 0.0, 1.0])
        dot = np.clip(np.dot(z, d), -1.0, 1.0)
        if abs(dot - 1.0) < 1e-8:
            return [0, 0, 0, 1]
        if abs(dot + 1.0) < 1e-8:
            return [1, 0, 0, 0]  # 180° about X
        axis = np.cross(z, d)
        axis = axis / np.linalg.norm(axis)
        angle = math.acos(dot)
        # Axis-angle → quaternion
        s = math.sin(angle / 2)
        return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2)]

    def _dummy_sim_step(self, n):
        for _ in range(n):
            self.bullet.stepSimulation()

    def set_marker(self, pos, color=None):
        """Place / move a translucent sphere marker in the scene."""
        color = color or [1, 0, 0, 0.5]
        if self._marker_id is None:
            vis = self.bullet.createVisualShape(
                self.bullet.GEOM_SPHERE, radius=0.008, rgbaColor=color,
            )
            self._marker_id = self.bullet.createMultiBody(
                baseMass=0, baseVisualShapeIndex=vis,
                basePosition=pos, baseOrientation=[0, 0, 0, 1],
            )
        else:
            self.bullet.resetBasePositionAndOrientation(
                self._marker_id, pos, [0, 0, 0, 1],
            )
        self._dummy_sim_step(1)

    def wait(self, sec):
        for _ in range(1 + int(sec / self._sim_dt)):
            self.bullet.stepSimulation()

    # ==================================================================
    # Contact detection
    # ==================================================================
    def is_in_contact(self, obj_id):
        """Check if any tube body overlaps with obj_id (AABB test)."""
        for body in self._tube_bodies:
            aabb1 = self.bullet.getAABB(body)
            aabb2 = self.bullet.getAABB(obj_id)
            overlap = all(
                aabb1[0][k] <= aabb2[1][k] and aabb1[1][k] >= aabb2[0][k]
                for k in range(3)
            )
            if overlap:
                return True
        return False

    def is_tip_in_contact(self, obj_id):
        """Check if the tip sphere overlaps obj_id."""
        aabb1 = self.bullet.getAABB(self._tube_bodies[-1])
        aabb2 = self.bullet.getAABB(obj_id)
        return all(
            aabb1[0][k] <= aabb2[1][k] and aabb1[1][k] >= aabb2[0][k]
            for k in range(3)
        )

    # convenience
    def add_cube(self, pos, size=None, mass=0.1, color=None):
        size = size or [0.02, 0.02, 0.02]
        color = color or [1, 1, 0, 1]
        col = self.bullet.createCollisionShape(
            self.bullet.GEOM_BOX,
            halfExtents=[s / 2 for s in size],
        )
        vis = self.bullet.createVisualShape(
            self.bullet.GEOM_BOX,
            halfExtents=[s / 2 for s in size],
            rgbaColor=color,
        )
        return self.bullet.createMultiBody(mass, col, vis, pos, [0, 0, 0, 1])
