"""
GESRsim Demo — Tendon-driven endoscope navigating a simple GI-tract tunnel.

This script demonstrates:
  1) Creating the endoscope (2-segment, 4-tendon-per-segment)
  2) Building a simplified GI tract (S-shaped tunnel with polyps)
  3) Steering the endoscope tip along a preset sinusoidal tendon trajectory
  4) 3D camera controller for real-time navigation

Run from the project root:
    python3 -m scripts.GESRsim_endoscope_demo
"""

import numpy as np
import time

from GESRsim.endoscope_env import EndoscopeEnv
from GESRsim.gi_tract_scene import GITractScene
from GESRsim.camera_controller import CameraController


def main():
    # ------------------------------------------------------------------
    # 1.  Create the endoscope environment
    # ------------------------------------------------------------------
    endoscope = EndoscopeEnv(
        num_segments=2,
        segment_length=60e-3,        # 60 mm per segment
        tendon_offset=4.0e-3,        # 4 mm tendon offset
        outer_radius=5.0e-3,         # 5 mm tube radius
        num_cylinders=40,
        body_color=[0.75, 0.75, 0.78, 1.0],   # metallic silver
        tip_color=[0.1, 0.1, 0.1, 1.0],       # dark tip
        camera_enabled=True,
        gui=True,
    )

    # ------------------------------------------------------------------
    # 2.  Build a GI-tract tunnel around the endoscope's workspace
    # ------------------------------------------------------------------
    tract = GITractScene(
        endoscope.bullet,
        inner_radius=0.018,      # 18 mm lumen
        wall_thickness=0.002,
        ring_spacing=0.010,
        wall_color=[0.88, 0.58, 0.58, 0.35],   # pinkish, semi-transparent
    )
    tract.build_simple_tunnel(origin=np.array([0.0, 0.0, 0.12]))

    # ------------------------------------------------------------------
    # 3.  3D Camera controller — navigate with keyboard
    # ------------------------------------------------------------------
    cam = CameraController(
        endoscope.bullet,
        distance=0.25,
        yaw=45.0,
        pitch=-25.0,
        target=[0.0, 0.15, 0.12],
    )

    # ------------------------------------------------------------------
    # 4.  Run a sinusoidal tendon-actuation demo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  GESRsim — Tendon-Driven Endoscope Demo")
    print("  2-segment, 4 tendons/segment, Cosserat-rod FK")
    print("=" * 60 + "\n")
    print("3D Camera controls (focus PyBullet window):")
    print("  WASD      : pan camera")
    print("  R / F     : move up / down")
    print("  Arrows    : orbit (yaw / pitch)")
    print("  Z / X     : zoom in / out")
    print("  H         : reset camera")
    print("  P         : print camera state\n")
    print("Running sinusoidal tendon actuation for 20 s ...")
    print("Press Ctrl+C to stop.\n")

    dt = 0.005
    t_final = 20.0
    gt = 0.0

    # Base position & orientation (endoscope pointing along Y+)
    base_pos = np.array([0.0, 0.0, 0.12])
    base_ori = np.array([np.pi / 2, 0.0, 0.0])    # tilted to point along Y+

    # Tendon action buffer: (2 segments × 4 tendons) = 8 values
    tendon_action = np.zeros(8)

    try:
        while gt < t_final:
            # --- Segment 1: slow sinusoidal bending ---
            amp1 = 0.003   # 3 mm max tendon pull
            w1 = 2.0 * np.pi / 8.0   # period = 8 s
            tendon_action[0] =  amp1 * np.sin(w1 * gt)        # tendon 1
            tendon_action[1] =  amp1 * np.cos(w1 * gt)        # tendon 2
            tendon_action[2] = -amp1 * np.sin(w1 * gt)        # tendon 3 (antagonist)
            tendon_action[3] = -amp1 * np.cos(w1 * gt)        # tendon 4 (antagonist)

            # --- Segment 2: faster oscillation ---
            amp2 = 0.002
            w2 = 2.0 * np.pi / 5.0   # period = 5 s
            tendon_action[4] =  amp2 * np.sin(w2 * gt + np.pi / 4)
            tendon_action[5] =  amp2 * np.cos(w2 * gt + np.pi / 4)
            tendon_action[6] = -amp2 * np.sin(w2 * gt + np.pi / 4)
            tendon_action[7] = -amp2 * np.cos(w2 * gt + np.pi / 4)

            # Slowly insert the endoscope along Y+
            insertion_speed = 0.002   # 2 mm/s
            base_pos[1] = insertion_speed * gt

            tip_pos, tip_ori, sol = endoscope.move(
                tendon_action, base_pos=base_pos, base_ori_euler=base_ori,
            )

            # Update 3D camera controller (poll keyboard)
            cam.update()

            if int(gt * 100) % 50 == 0:
                print(f"  t = {gt:5.1f} s  |  tip = ({tip_pos[0]:+.4f}, {tip_pos[1]:+.4f}, {tip_pos[2]:+.4f})")

            gt += dt
            time.sleep(dt * 0.5)   # slow down slightly for visualisation

    except KeyboardInterrupt:
        print("\nStopped by user.")

    print("\nDemo complete.  Close the PyBullet window to exit.")
    endoscope.wait(5)


if __name__ == "__main__":
    main()
