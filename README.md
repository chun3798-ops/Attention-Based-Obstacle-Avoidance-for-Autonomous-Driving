# Attention-Based Obstacle Avoidance for Autonomous Driving

An attention-based 2D simulation in which a vehicle avoids randomly moving obstacles. Built in Python, it trains an attention policy via imitation learning in two ways: outputting a steering angle directly, or outputting a path. Path output is implemented with two approaches: a 5th-order polynomial and a cubic spline curve.

## Contents

| Folder | Method | Status |
|---|---|---|
| `Steering Output/` | Attention policy outputs steering angle and acceleration directly | Available |
| `Quintic Polynomial/` | Attention policy outputs a target point and target speed, joined to the current state by a 5th-order polynomial path | Available |
| `Spline Curve/` | Attention policy outputs 5 knot points, joined by a cubic spline path | Available |

## Steering Output

- `nav_sim.py` - 2D simulator (random static/moving obstacles, kinematic bicycle model)
- `attn_policy.py` - self-attention policy; tokens `[ego, goal, obstacles...]` -> `[steering, acceleration]`
- `train.py` - imitation learning from an artificial potential field (APF) teacher, with DAgger rounds
- `save_gif.py` - runs the policy and saves `attn_result.gif`
- `policy.pt` - trained weights

### Requirements

```
pip install numpy torch matplotlib pillow
```

### Usage

```
cd "Steering Output"
python train.py       # train and save policy.pt
python attn_policy.py # run the animation with the trained policy
python save_gif.py    # save attn_result.gif
```

## Quintic Polynomial

At every step the attention policy outputs a target point (dx, dy) and a target speed ratio. A 5th-order polynomial that satisfies position, velocity and acceleration boundary conditions connects the current state to that target, and a fixed-gain controller follows a point slightly ahead on the curve.

- `nav_sim.py`, `attn_policy.py` - same simulator and attention backbone as `Steering Output/`
- `path2_quintic.py` - quintic polynomial boundary-condition solver (position and velocity)
- `path2_train.py` - imitation learning (BC + DAgger) from an APF teacher; saves `path2_policy.pt` and a GIF
- `path2_save_gif.py` - renders `path2_result.gif` from trained weights
- `evaluate.py` - success rate over 200 random maps
- `path2_policy.pt` - trained weights

```
cd "Quintic Polynomial"
python path2_train.py     # train, save path2_policy.pt and path2_result.gif
python path2_save_gif.py  # re-render the GIF only
python evaluate.py        # success rate
```

## Spline Curve

The attention policy outputs 5 points (x, y) at fixed knot times (t = 0.2 ... 1.0). Together with the current position they are joined by a natural cubic spline (implemented with NumPy only), and a fixed-gain controller follows a point on the spline.

- `nav_sim.py`, `attn_policy.py` - same simulator and attention backbone as `Steering Output/`
- `path3_spline.py` - natural cubic spline path generator (position and velocity)
- `path3_train.py` - imitation learning (BC + DAgger) from an APF teacher that provides the knot points; saves `path3_policy.pt` and a GIF
- `path3_save_gif.py` - renders `path3_result.gif` from trained weights
- `evaluate.py` - success rate over 500 random maps
- `path3_policy.pt` - trained weights

```
cd "Spline Curve"
python path3_train.py     # train, save path3_policy.pt and path3_result.gif
python path3_save_gif.py  # re-render the GIF only
python evaluate.py        # success rate
```
