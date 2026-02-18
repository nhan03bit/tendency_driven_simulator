"""
GESRsim PD Control — Jacobian-based tip tracking for the tendon-driven
endoscope.

The endoscope tip follows a reference trajectory inside the GI tract using
numerical Jacobian + PD control of the 4-tendon-per-segment actuation.

Run from the project root:
    python -m scripts.GESRsim_endoscope_PD_control
"""

import numpy as np
import time

from GESRsim.endoscope_env import EndoscopeEnv
from GESRsim.gi_tract_scene import GITractScene


# ======================================================================
# Numerical Jacobian  J = d(tip) / d(tendon_actions)
# ======================================================================
def numerical_jacobian(env, actions, eps=1e-5):
    """3×N Jacobian of the tip position w.r.t. tendon actions."""
    actions = np.asarray(actions).ravel()
    n = len(actions)
    p0 = env.calc_tip_pos(actions)
    J = np.zeros((3, n))
    for i in range(n):
        da = np.zeros(n)
        da[i] = eps
        p_plus = env.calc_tip_pos(actions + da)
        p_minus = env.calc_tip_pos(actions - da)
        J[:, i] = (p_plus - p_minus) / (2 * eps)
    return J


# ======================================================================
# Reference trajectories (in local endoscope workspace)
# ======================================================================
def get_reference(t, x0, traj_name="Circle"):
    if traj_name == "Circle":
        T = 15.0
        w = 2 * np.pi / T
        r = 0.015
        xd = x0 + np.array([r * np.sin(w * t), r * np.cos(w * t), 0.002 * t])
        xd_dot = np.array([r * w * np.cos(w * t), -r * w * np.sin(w * t), 0.002])
    elif traj_name == "Helix":
        T = 12.0
        w = 2 * np.pi / T
        r = 0.012
        xd = x0 + np.array([r * np.sin(w * t), r * np.cos(w * t), 0.003 * t])
        xd_dot = np.array([r * w * np.cos(w * t), -r * w * np.sin(w * t), 0.003])
    elif traj_name == "Eight":
        T = 18.0
        w = 2 * np.pi / T
        A = 0.012
        xd = x0 + np.array([A * np.sin(w * t), A * np.sin(w / 2 * t), 0.001 * t])
        xd_dot = np.array([A * w * np.cos(w * t), A * w / 2 * np.cos(w / 2 * t), 0.001])
    elif traj_name == "Straight":
        speed = 0.005  # 5 mm/s
        xd = x0 + np.array([0, 0, speed * t])
        xd_dot = np.array([0, 0, speed])
    else:
        xd = x0.copy()
        xd_dot = np.zeros(3)
    return xd, xd_dot


# ======================================================================
# Main
# ======================================================================
def main():
    # ----- Config -----
    traj_name = "Circle"
    t_final = 25.0
    ts = 0.005
    save_log = True

    # ----- Create endoscope -----
    endoscope = EndoscopeEnv(
        num_segments=2,
        segment_length=60e-3,
        tendon_offset=4.0e-3,
        outer_radius=5.0e-3,
        num_cylinders=40,
        camera_enabled=True,
        gui=True,
    )

    # ----- GI tract -----
    tract = GITractScene(
        endoscope.bullet,
        inner_radius=0.020,
        wall_thickness=0.002,
        ring_spacing=0.010,
        wall_color=[0.88, 0.58, 0.58, 0.3],
    )
    tract.build_simple_tunnel(origin=np.array([0.0, 0.0, 0.01]))

    # ----- Initial pose -----
    base_pos = np.array([0.0, 0.0, 0.015])
    base_ori = np.array([0.0, 0.0, 0.0])
    tendon_action = np.zeros(8)

    tip_pos, _, _ = endoscope.move(tendon_action, base_pos, base_ori)
    x0 = tip_pos.copy()

    # PD gains
    Kp = 4.0 * np.eye(3)

    print("\n" + "=" * 60)
    print(f"  GESRsim — PD Tip Tracking  ({traj_name})")
    print("=" * 60 + "\n")

    gt = 0.0
    prev_pose = x0.copy()
    log_data = []

    # Draw reference trajectory preview
    for i in range(int(t_final / (ts * 10))):
        t_pre = i * ts * 10
        xd, _ = get_reference(t_pre, x0, traj_name)
        endoscope.bullet.addUserDebugLine(prev_pose.tolist(), xd.tolist(), [0, 0, 0.4], 2, 0)
        prev_pose = xd
    prev_pose = x0.copy()

    try:
        while gt < t_final:
            t_now = time.time()

            # Reference
            xd, xd_dot = get_reference(gt, x0, traj_name)

            # Current tip
            tip_pos = endoscope.calc_tip_pos(tendon_action)

            # Error
            err = xd - tip_pos
            err = np.clip(err, -0.02, 0.02)

            # Jacobian
            J = numerical_jacobian(endoscope, tendon_action)

            # Pseudo-inverse control
            J_pinv = np.linalg.pinv(J)
            u = J_pinv @ (xd_dot + Kp @ err)

            # Update tendon actions
            tendon_action += u * ts
            tendon_action = np.clip(tendon_action, -0.008, 0.008)

            # Slowly insert
            base_pos[2] = 0.015 + 0.001 * gt

            # Move
            tip_pos, tip_ori, sol = endoscope.move(
                tendon_action, base_pos=base_pos, base_ori_euler=base_ori,
            )

            # Visualise actual path
            if int(gt * 100) % 10 == 0:
                endoscope.bullet.addUserDebugLine(
                    prev_pose.tolist(), tip_pos.tolist(), [1, 0, 0.3], 3, 0,
                )
                prev_pose = tip_pos.copy()

            # Log
            if save_log:
                log_data.append(np.concatenate([[gt], tip_pos, xd, tendon_action]))

            if int(gt * 100) % 50 == 0:
                e_norm = np.linalg.norm(err) * 1000  # mm
                print(f"  t={gt:5.1f}s  err={e_norm:5.2f}mm  tip=({tip_pos[0]:+.4f},{tip_pos[1]:+.4f},{tip_pos[2]:+.4f})")

            gt += ts
            elapsed = time.time() - t_now
            if elapsed < ts:
                time.sleep(ts - elapsed)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    # Save log
    if save_log and log_data:
        import os
        os.makedirs("scripts/logs", exist_ok=True)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        fname = f"scripts/logs/gesrsim_pd_{traj_name}_{timestr}.dat"
        header = "#t,tip_x,tip_y,tip_z,ref_x,ref_y,ref_z,t1..t8"
        np.savetxt(fname, np.array(log_data), fmt="%.6f", header=header)
        print(f"\nLog saved: {fname}")

    print("\nPD control demo complete.")
    endoscope.wait(5)


if __name__ == "__main__":
    main()
