"""
GESRsim Teleoperation — Keyboard control of the tendon-driven endoscope.

Controls
--------
  Arrow keys     : steer segment-1 (proximal)
  W / S          : steer segment-2 up / down
  A / D          : steer segment-2 left / right
  Q / E          : insert / retract the endoscope
  SPACE          : reset all tendon actions to zero
  V              : toggle endoscope camera view (OpenCV window)
  ESC            : quit

Run from the project root:
    python -m scripts.GESRsim_endoscope_teleop
"""

import numpy as np
import time
import threading
import cv2

from GESRsim.endoscope_env import EndoscopeEnv
from GESRsim.gi_tract_scene import GITractScene
from Keyboard.keyboardThread import KeyboardThread


def main():
    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------
    endoscope = EndoscopeEnv(
        num_segments=2,
        segment_length=60e-3,
        tendon_offset=4.0e-3,
        outer_radius=5.0e-3,
        num_cylinders=40,
        body_color=[0.75, 0.75, 0.78, 1.0],
        tip_color=[0.1, 0.1, 0.1, 1.0],
        camera_enabled=True,
        gui=True,
    )

    tract = GITractScene(
        endoscope.bullet,
        inner_radius=0.018,
        wall_thickness=0.002,
        ring_spacing=0.010,
        wall_color=[0.88, 0.58, 0.58, 0.35],
    )
    tract.build_simple_tunnel(origin=np.array([0.0, 0.0, 0.01]))

    # ------------------------------------------------------------------
    # Keyboard thread
    # ------------------------------------------------------------------
    key_lock = threading.Lock()
    key_thr = KeyboardThread(freq=30, lock=key_lock)
    get_key_thread = threading.Thread(target=key_thr.readkey, daemon=True)
    get_key_thread.start()

    print("\n" + "=" * 60)
    print("  GESRsim — Keyboard Teleoperation")
    print("  Arrows=seg1  W/S/A/D=seg2  Q/E=insert/retract")
    print("  SPACE=zero  V=cam view  ESC=quit")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    dt = 0.005
    tendon_step = 0.0003    # mm per keystroke
    insert_step = 0.001     # mm per keystroke

    base_pos = np.array([0.0, 0.0, 0.015])
    base_ori = np.array([0.0, 0.0, 0.0])
    tendon_action = np.zeros(8)
    show_cam = False

    try:
        while True:
            # Poll keyboard
            ux, uy, uz, key = key_thr.updateKeyInfo()

            # Map ux/uy to segment-1 tendons
            tendon_action[0] += uy * tendon_step       # up/down → tendon 1
            tendon_action[2] -= uy * tendon_step       # antagonist
            tendon_action[1] += ux * tendon_step       # left/right → tendon 2
            tendon_action[3] -= ux * tendon_step       # antagonist

            # Segment-2 via W/S/A/D
            if key == 'w' or key == 'W':
                tendon_action[4] += tendon_step
                tendon_action[6] -= tendon_step
            elif key == 's' or key == 'S':
                tendon_action[4] -= tendon_step
                tendon_action[6] += tendon_step
            elif key == 'a' or key == 'A':
                tendon_action[5] += tendon_step
                tendon_action[7] -= tendon_step
            elif key == 'd' or key == 'D':
                tendon_action[5] -= tendon_step
                tendon_action[7] += tendon_step

            # Insert / retract with Q / E
            if key == 'q' or key == 'Q':
                base_pos[2] += insert_step
            elif key == 'e' or key == 'E':
                base_pos[2] -= insert_step
                base_pos[2] = max(base_pos[2], 0.005)

            # Reset
            if key == ' ':
                tendon_action[:] = 0.0

            # Toggle camera view
            if key == 'v' or key == 'V':
                show_cam = not show_cam

            # Clamp tendon actions
            tendon_action = np.clip(tendon_action, -0.008, 0.008)

            # Step the simulation
            tip_pos, tip_ori, sol = endoscope.move(
                tendon_action, base_pos=base_pos, base_ori_euler=base_ori,
            )

            # Show endoscope camera if toggled
            if show_cam:
                rgb, depth = endoscope.capture_endoscope_image()
                if rgb is not None:
                    cv2.imshow("Endoscope Camera", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            else:
                cv2.destroyAllWindows()

            # Contact check with tract wall rings
            contact = endoscope.is_in_contact(tract._ring_bodies[0]) if tract._ring_bodies else False
            status = "CONTACT" if contact else "       "

            if int(time.time() * 10) % 5 == 0:
                print(
                    f"\r  tip=({tip_pos[0]:+.4f},{tip_pos[1]:+.4f},{tip_pos[2]:+.4f})  "
                    f"ins={base_pos[2]:.3f}m  {status}",
                    end="",
                )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n\nTeleoperation ended.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
