"""
2D 자율주행 네비게이션 시뮬레이터 (랜덤 + 움직이는 장애물)
====================================================
출발지 -> 목적지, 장애물 회피 환경.
reset() 마다 장애물이 랜덤하게 새로 배치됨 (정지/이동 섞임).

★ 네가 할 일: 아래 my_algorithm() 함수만 채우면 됨.

실행:  python nav_sim.py
필요:  numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


# ============================================================
# 1. 시나리오 / 랜덤 설정
# ============================================================
START = np.array([2.0, 2.0])         # 출발 (x, y)
GOAL  = np.array([45.0, 45.0])       # 목적지 (x, y)
WORLD_SIZE = 50.0

RANDOMIZE   = True          # True = 매번 랜덤 배치 / False = 아래 고정 배치
SEED        = None          # None = 매번 다름 / 정수 = 재현 가능 (예: 42)

N_OBSTACLES     = 6         # 장애물 개수
RADIUS_RANGE    = (1.3, 2.7)   # 반지름 범위 (기존의 약 2/3로 축소)
SPEED_RANGE     = (1.0, 3.0)   # 움직일 때 속도 범위 (m/s)
MOVING_FRACTION = 0.6          # 움직이는 장애물 비율 (0~1)
SPAWN_MARGIN    = 4.0          # 출발/목적지에서 띄우는 최소 여유

# RANDOMIZE=False 일 때 쓸 고정 배치: [x, y, r, vx, vy]
OBSTACLES_FIXED = np.array([
    [15.0, 12.0, 3.0,  0.0,  2.5],
    [25.0, 28.0, 4.0, -2.0,  0.0],
    [35.0, 20.0, 2.5,  0.0,  0.0],
])


# ============================================================
# 2. 차량 / 시뮬레이션 파라미터
# ============================================================
DT          = 0.1
WHEELBASE   = 2.0
V_MAX       = 8.0
V_MIN       = 0.0
STEER_MAX   = np.deg2rad(35)
ACCEL_MAX   = 3.0
GOAL_RADIUS = 1.5
CAR_RADIUS  = 1.0
MAX_STEPS   = 1000


# ============================================================
# 3. 환경 (완성됨 — 건드릴 필요 없음)
# ============================================================
class NavEnv:
    def __init__(self):
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(SEED if seed is None else seed)
        x, y = START
        d = GOAL - START
        heading = np.arctan2(d[1], d[0])
        self.state = np.array([x, y, heading, 0.0])
        if RANDOMIZE:
            self.obstacles = self._spawn_obstacles()
        else:
            self.obstacles = OBSTACLES_FIXED.copy().astype(float)
        self.steps = 0
        self.trail = [self.state[:2].copy()]
        return self._obs()

    def _spawn_obstacles(self):
        rng = self.rng
        obs, attempts = [], 0
        while len(obs) < N_OBSTACLES and attempts < 2000:
            attempts += 1
            r = rng.uniform(*RADIUS_RANGE)
            x = rng.uniform(r, WORLD_SIZE - r)
            y = rng.uniform(r, WORLD_SIZE - r)
            if np.hypot(x - START[0], y - START[1]) < r + SPAWN_MARGIN:
                continue
            if np.hypot(x - GOAL[0],  y - GOAL[1])  < r + SPAWN_MARGIN:
                continue
            if any(np.hypot(x - ox, y - oy) < r + orad + 1.0
                   for ox, oy, orad, _, _ in obs):
                continue
            obs.append([x, y, r, 0.0, 0.0])      # 일단 전부 정지로 배치
        obs = np.array(obs, dtype=float)
        # 그 중 2~3개만 움직이게 (속도 부여)
        if len(obs) > 0:
            n_move = min(len(obs), int(rng.integers(2, 4)))   # 2 또는 3개
            for i in rng.choice(len(obs), n_move, replace=False):
                ang = rng.uniform(0, 2 * np.pi)
                spd = rng.uniform(*SPEED_RANGE)
                obs[i, 3] = spd * np.cos(ang)
                obs[i, 4] = spd * np.sin(ang)
        return obs

    def _move_obstacles(self):
        o = self.obstacles
        o[:, 0] += o[:, 3] * DT
        o[:, 1] += o[:, 4] * DT
        for i in range(len(o)):
            r = o[i, 2]
            if o[i, 0] < r or o[i, 0] > WORLD_SIZE - r:
                o[i, 3] *= -1; o[i, 0] = np.clip(o[i, 0], r, WORLD_SIZE - r)
            if o[i, 1] < r or o[i, 1] > WORLD_SIZE - r:
                o[i, 4] *= -1; o[i, 1] = np.clip(o[i, 1], r, WORLD_SIZE - r)

    def _obs(self):
        x, y, h, v = self.state
        ego = np.array([x, y, h, v])
        goal_rel = GOAL - np.array([x, y])
        obs_abs = self.obstacles.copy()        # [N,5] (x,y,r,vx,vy)
        obs_rel = obs_abs.copy()
        obs_rel[:, 0] -= x
        obs_rel[:, 1] -= y
        return {
            "ego":           ego,
            "goal":          GOAL.copy(),
            "goal_rel":      goal_rel,
            "obstacles":     obs_abs,
            "obstacles_rel": obs_rel,
        }

    def step(self, action):
        steer, accel = action
        steer = float(np.clip(steer, -STEER_MAX, STEER_MAX))
        accel = float(np.clip(accel, -ACCEL_MAX, ACCEL_MAX))
        x, y, h, v = self.state
        x += v * np.cos(h) * DT
        y += v * np.sin(h) * DT
        h += v / WHEELBASE * np.tan(steer) * DT
        v  = float(np.clip(v + accel * DT, V_MIN, V_MAX))
        self.state = np.array([x, y, h, v])
        self._move_obstacles()
        self.steps += 1
        self.trail.append(np.array([x, y]))
        dist_goal = float(np.linalg.norm(GOAL - np.array([x, y])))
        reached   = dist_goal < GOAL_RADIUS
        collided  = self._check_collision(x, y)
        out       = not (0 <= x <= WORLD_SIZE and 0 <= y <= WORLD_SIZE)
        done      = reached or collided or out or self.steps >= MAX_STEPS
        info = {"reached": reached, "collided": collided,
                "out_of_bounds": out, "dist_goal": dist_goal}
        return self._obs(), done, info

    def _check_collision(self, x, y):
        for ox, oy, r, vx, vy in self.obstacles:
            if np.hypot(x - ox, y - oy) < r + CAR_RADIUS:
                return True
        return False


# ============================================================
# 4. ★★★ 여기가 네가 짤 부분 ★★★
# ============================================================
def my_algorithm(obs):
    """
    obs["ego"]            : [x, y, heading, v]
    obs["goal_rel"]       : [dx, dy]
    obs["obstacles_rel"]  : [N,5] (dx, dy, r, vx, vy)
    출력: np.array([조향각(rad), 가속도(m/s^2)])

    아래는 더미(목적지 직진, 회피 X). 지우고 네 회피 알고리즘을 넣어.
    ※ 랜덤 배치라 알고리즘이 '특정 맵 암기'가 아니라
      '일반적 회피'를 하는지 검증하기 좋음.
    """
    x, y, heading, v = obs["ego"]
    dx, dy = obs["goal_rel"]
    target = np.arctan2(dy, dx)
    err = np.arctan2(np.sin(target - heading), np.cos(target - heading))
    return np.array([2.0 * err, 1.0])


# ============================================================
# 5. 시각화 + 실행
# ============================================================
def run(animate=True):
    env = NavEnv()
    obs = env._obs()
    if not animate:
        done, info = False, {}
        while not done:
            obs, done, info = env.step(my_algorithm(obs))
        return env, info

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, WORLD_SIZE); ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    obstacle_patches = []
    for ox, oy, r, vx, vy in env.obstacles:
        c = Circle((ox, oy), r, color="#d9534f", alpha=0.6)
        ax.add_patch(c); obstacle_patches.append(c)
    ax.plot(*GOAL,  "*", color="gold",  markersize=22, markeredgecolor="k", label="Goal")
    ax.plot(*START, "o", color="green", markersize=8,  label="Start")
    (trail_line,) = ax.plot([], [], "-", color="#0275d8", lw=1.5)
    (car_dot,)    = ax.plot([], [], "^", color="#0275d8", markersize=12)
    title = ax.set_title("")
    state = {"obs": obs, "done": False, "info": {}}

    def update(_frame):
        if not state["done"]:
            state["obs"], state["done"], state["info"] = env.step(my_algorithm(state["obs"]))
        for c, (ox, oy, r, vx, vy) in zip(obstacle_patches, env.obstacles):
            c.center = (ox, oy)
        trail = np.array(env.trail)
        trail_line.set_data(trail[:, 0], trail[:, 1])
        x, y, h, v = env.state
        car_dot.set_data([x], [y])
        i = state["info"]
        if state["done"]:
            if   i.get("reached"):       msg = "도착! ✅"
            elif i.get("collided"):      msg = "충돌 ❌"
            elif i.get("out_of_bounds"): msg = "맵 이탈 ❌"
            else:                        msg = "시간 초과 ⏱"
            title.set_text(f"{msg}   (steps={env.steps})")
        else:
            title.set_text(f"step={env.steps}   v={v:.1f}   "
                           f"dist={i.get('dist_goal', float('nan')):.1f}")
        return (*obstacle_patches, trail_line, car_dot, title)

    ani = FuncAnimation(fig, update, frames=MAX_STEPS,
                        interval=30, blit=False, repeat=False)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    return env, state["info"], ani


if __name__ == "__main__":
    run(animate=True)
