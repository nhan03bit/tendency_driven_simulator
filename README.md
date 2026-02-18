
# GESRsim — Gastrointestinal Endoscopic Surgical Robot Simulator

**GESRsim** is a Python simulator for tendon-driven continuum endoscopes, inspired by the GESRsim concept and built on top of [SoftManiSim](https://github.com/MohammadKasaei/SoftManiSim). It features real-time Cosserat rod kinematics, PyBullet-based physics and rendering, a modular GI tract builder, and a 3D camera controller for interactive navigation and teleoperation research.


## Features

- Multi-segment tendon-driven endoscope (4 tendons/segment, antagonistic pairs)
- Cosserat rod ODE solver for real-time backbone kinematics
- Cylinder-chain rendering for smooth, realistic endoscope visuals
- Tip-mounted endoscope camera (RGB + depth, 120° FOV)
- Modular GI tract scene builder (straight, curved, S-shaped, colon layout, polyps)
- 3D camera controller (WASD, arrows, zoom, reset, print)
- Jacobian-based PD tip tracking (closed-loop, multiple trajectories)
- Keyboard teleoperation (real-time tendon steering, insertion/retraction)


## Project Structure

```
GESRsim/
  endoscope_ode.py          # Cosserat rod FK (4-tendon per segment)
  endoscope_env.py          # PyBullet environment (cylinder rendering, camera, contacts)
  gi_tract_scene.py         # GI tract tube builder + polyp placement
  camera_controller.py      # Real-time 3D camera navigation

scripts/
  GESRsim_endoscope_demo.py        # Sinusoidal tendon actuation demo
  GESRsim_endoscope_teleop.py      # Keyboard teleoperation
  GESRsim_endoscope_PD_control.py  # Jacobian PD tip tracking
```


## Installation

### Prerequisites

- Python 3.10 (recommended)
- Conda or virtualenv

### Setup

Clone the repository and set up the environment:

```bash
git clone https://github.com/nhan03bit/tendency_driven_simulator.git
cd tendency_driven_simulator
conda create -n gesrsim python=3.10 -y
conda activate gesrsim
chmod +x install_dependencies.sh
./install_dependencies.sh
```


## Quick Start

### Endoscope Demo (auto sinusoidal steering + GI tract)

```bash
python3 -m scripts.GESRsim_endoscope_demo
```

### Keyboard Teleoperation

```bash
python3 -m scripts.GESRsim_endoscope_teleop
```

**Controls:**
| Key | Action |
|---|---|
| Arrow keys | Steer segment 1 (proximal) |
| W / S / A / D | Steer segment 2 (distal) |
| Q / E | Insert / retract endoscope |
| SPACE | Reset all tendons to zero |
| V | Toggle endoscope camera view |

### PD Tip Tracking

```bash
python3 -m scripts.GESRsim_endoscope_PD_control
```

Tracks Circle, Helix, Figure-8, or Straight trajectories using numerical Jacobian + PD control.

### 3D Camera Navigation

Available in all scripts. Focus the PyBullet window and use:

| Key | Action |
|---|---|
| W / A / S / D | Pan camera |
| R / F | Move up / down |
| Arrow keys | Orbit (yaw / pitch) |
| Z / X | Zoom in / out |
| H | Reset camera |
| P | Print camera state |


## API Reference

### EndoscopeODE — Cosserat Rod Kinematics

| Method | Description |
|---|---|
| forward(actions) | Full FK for all segments. Input: (N_seg, 4) tendon displacements. Returns backbone shape. |
| forward_simple(actions) | SoftManiSim-compatible 3-per-segment format. |
| tip_position(actions) | Returns (x, y, z) of the tip. |
| tip_jacobian(actions) | Numerical Jacobian (3, 4*N_seg). |
| tendon_to_curvature(deltas) | Converts 4 tendon pulls → (l, ux, uy) curvatures. |

### EndoscopeEnv — PyBullet Environment

| Method | Description |
|---|---|
| move(tendon_actions, base_pos, base_ori_euler) | FK + update visuals. Returns tip position, orientation, and solution. |
| move_simple(actions, ...) | Same but with 3-per-segment format. |
| calc_tip_pos(tendon_actions) | Tip position without visual update. |
| capture_endoscope_image() | RGB + depth from the tip camera. |
| is_in_contact(obj_id) | AABB contact check (any body). |
| is_tip_in_contact(obj_id) | AABB contact check (tip only). |

### GITractScene — GI Tract Builder

| Method | Description |
|---|---|
| build_straight(start, direction, length) | Straight tube segment. |
| build_curve(start, dir, angle, radius) | Constant-curvature bend. |
| build_s_curve(...) | Two opposite bends. |
| build_colon_layout(origin) | Preset colon anatomy. |
| build_simple_tunnel(origin) | Quick S-shaped test tunnel. |
| add_polyp(position, radius) | Place a single polyp sphere. |
| add_polyps_along_wall(n) | Randomly place polyps on the inner wall. |

### CameraController — 3D Viewport Navigation

| Method | Description |
|---|---|
| update() | Poll keyboard and move camera. Call once per frame. |
| set_pose(distance, yaw, pitch, target) | Programmatic camera positioning. |
| follow(position, smooth) | Smooth tracking of a moving object. |


## Architecture

```
EndoscopeODE  (Cosserat rod, 4 tendons/segment)
  │
  ▼
EndoscopeEnv  (PyBullet cylinder rendering + tip camera)
  │
  ├── GITractScene  (tubular walls + polyps)
  ├── CameraController  (3D viewport navigation)
  │
  └── Scripts:
     ├── demo       (auto sinusoidal steering)
     ├── teleop     (keyboard control)
     └── PD_control (Jacobian tip tracking)
```


## References

- **SoftManiSim** — [Kasaei et al., CoRL 2024](https://openreview.net/pdf?id=ovjxugn9Q2): Fast simulation framework for multi-segment continuum manipulators using Cosserat rod theory.
- **GESRsim** — Gastrointestinal Endoscopic Surgical Robot Simulator concept, adapted for tendon-driven continuum robots.

## License

This project is licensed under the MIT License.
