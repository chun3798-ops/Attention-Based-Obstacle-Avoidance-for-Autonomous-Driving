# Attention-Based Obstacle Avoidance for Autonomous Driving

An attention-based 2D simulation in which a vehicle avoids randomly moving obstacles. Built in Python, it trains an attention policy via imitation learning in two ways: outputting a steering angle directly, or outputting a path. Path output is implemented with two approaches: a 5th-order polynomial and a cubic spline curve.

![result](Steering%20Output/attn_result.gif)

## Contents

| Folder | Method | Status |
|---|---|---|
| `Steering Output/` | Attention policy outputs steering angle and acceleration directly | Available |
| 5th-order polynomial path | Attention policy outputs a polynomial path | Planned |
| Cubic spline path | Attention policy outputs a cubic spline path | Planned |

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
