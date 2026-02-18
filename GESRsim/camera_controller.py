"""
CameraController — Real-time 3D camera navigation for any PyBullet scene
in the SoftManiSim framework.

Controls (while the PyBullet window is focused):
─────────────────────────────────────────────────
  WASD          : Pan the camera target (forward / left / back / right)
  R / F         : Move target up / down
  Arrow Left/Right : Orbit yaw (rotate around target)
  Arrow Up/Down    : Orbit pitch (tilt)
  Z / X         : Zoom in / out
  Home (or H)   : Reset camera to initial pose
  P             : Print current camera state to console

Usage
-----
    from GESRsim.camera_controller import CameraController

    # After creating your PyBullet environment:
    cam = CameraController(bullet_client)

    # Inside your simulation loop:
    while running:
        cam.update()          # poll keys & move camera
        # ... rest of sim ...
"""

import math
import numpy as np


# PyBullet key constants (ASCII codes & special keys)
_KEY_W = ord('w')
_KEY_A = ord('a')
_KEY_S = ord('s')
_KEY_D = ord('d')
_KEY_R = ord('r')
_KEY_F = ord('f')
_KEY_Z = ord('z')
_KEY_X = ord('x')
_KEY_H = ord('h')
_KEY_P = ord('p')

# Arrow keys in PyBullet use special codes (>= 65000)
_KEY_LEFT  = 65295
_KEY_RIGHT = 65296
_KEY_UP    = 65297
_KEY_DOWN  = 65298


class CameraController:
    """
    Interactive 3D camera navigation for PyBullet's debug visualiser.

    Parameters
    ----------
    bullet : pybullet module / client
        The PyBullet instance running the GUI.
    distance : float
        Initial camera distance from the target (metres).
    yaw : float
        Initial yaw angle (degrees).
    pitch : float
        Initial pitch angle (degrees).  Negative = looking down.
    target : array-like, shape (3,)
        Initial camera look-at position [x, y, z].
    pan_speed : float
        Translation speed (m per update call).
    orbit_speed : float
        Rotation speed (degrees per update call).
    zoom_speed : float
        Zoom speed (m per update call).
    min_distance : float
        Minimum allowed zoom distance.
    max_distance : float
        Maximum allowed zoom distance.
    """

    def __init__(
        self,
        bullet,
        distance: float = 0.35,
        yaw: float = 45.0,
        pitch: float = -25.0,
        target=None,
        pan_speed: float = 0.005,
        orbit_speed: float = 2.0,
        zoom_speed: float = 0.01,
        min_distance: float = 0.05,
        max_distance: float = 5.0,
    ):
        self.bullet = bullet
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.target = np.array(target if target is not None else [0.0, 0.0, 0.1])

        self.pan_speed = pan_speed
        self.orbit_speed = orbit_speed
        self.zoom_speed = zoom_speed
        self.min_distance = min_distance
        self.max_distance = max_distance

        # Save initial state for reset
        self._init_distance = self.distance
        self._init_yaw = self.yaw
        self._init_pitch = self.pitch
        self._init_target = self.target.copy()

        # Apply the initial camera pose
        self._apply()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self):
        """
        Poll PyBullet keyboard events and update the camera.
        Call this once per simulation step / frame.
        """
        keys = self.bullet.getKeyboardEvents()
        if not keys:
            return

        # Helper: check if a key is currently held (state & 3 covers
        # KEY_IS_DOWN=1 and KEY_WAS_TRIGGERED=2)
        def held(k):
            return keys.get(k, 0) & 3

        # --- Orbit (yaw / pitch) ---
        if held(_KEY_LEFT):
            self.yaw -= self.orbit_speed
        if held(_KEY_RIGHT):
            self.yaw += self.orbit_speed
        if held(_KEY_UP):
            self.pitch = min(self.pitch + self.orbit_speed, -5.0)
        if held(_KEY_DOWN):
            self.pitch = max(self.pitch - self.orbit_speed, -89.0)

        # --- Pan (WASD in world-horizontal plane relative to yaw) ---
        yaw_rad = math.radians(self.yaw)
        fwd = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
        right = np.array([math.sin(yaw_rad), -math.cos(yaw_rad), 0.0])

        if held(_KEY_W):
            self.target += fwd * self.pan_speed
        if held(_KEY_S):
            self.target -= fwd * self.pan_speed
        if held(_KEY_A):
            self.target -= right * self.pan_speed
        if held(_KEY_D):
            self.target += right * self.pan_speed

        # --- Vertical ---
        if held(_KEY_R):
            self.target[2] += self.pan_speed
        if held(_KEY_F):
            self.target[2] -= self.pan_speed

        # --- Zoom ---
        if held(_KEY_Z):
            self.distance = max(self.distance - self.zoom_speed, self.min_distance)
        if held(_KEY_X):
            self.distance = min(self.distance + self.zoom_speed, self.max_distance)

        # --- Reset ---
        if held(_KEY_H):
            self.distance = self._init_distance
            self.yaw = self._init_yaw
            self.pitch = self._init_pitch
            self.target = self._init_target.copy()

        # --- Print state ---
        if keys.get(_KEY_P, 0) == 2:   # triggered only (not held)
            print(
                f"[CameraController]  dist={self.distance:.3f}  "
                f"yaw={self.yaw:.1f}  pitch={self.pitch:.1f}  "
                f"target=({self.target[0]:.3f}, {self.target[1]:.3f}, {self.target[2]:.3f})"
            )

        self._apply()

    def set_pose(self, distance=None, yaw=None, pitch=None, target=None):
        """Programmatically set the camera pose."""
        if distance is not None:
            self.distance = distance
        if yaw is not None:
            self.yaw = yaw
        if pitch is not None:
            self.pitch = pitch
        if target is not None:
            self.target = np.asarray(target, dtype=float)
        self._apply()

    def follow(self, position, smooth=0.1):
        """
        Smoothly move the camera target toward *position*.
        Call each frame to follow a moving object.

        Parameters
        ----------
        position : (3,)  — world position to track
        smooth : float   — interpolation factor (0 = no move, 1 = snap)
        """
        self.target = (1.0 - smooth) * self.target + smooth * np.asarray(position)
        self._apply()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _apply(self):
        self.bullet.resetDebugVisualizerCamera(
            cameraDistance=self.distance,
            cameraYaw=self.yaw,
            cameraPitch=self.pitch,
            cameraTargetPosition=self.target.tolist(),
        )
