"""
path3_train.py
--------------
Level-3 경로 출력 방식:
  attention이 "시간 고정 매듭 5개"의 위치를 직접 출력 -> cubic spline으로 부드럽게 연결.

핵심 구조 (이전 단계와의 차이):
  - 출력: 이전(path2)은 [목표점(2) + 목표속도(1)] = 3개였음.
          이번은 [5개 점 x,y] = 10개. (속도는 spline 미분으로 자동)
  - 매듭 시간은 고정: t = 0.2, 0.4, 0.6, 0.8, 1.0 (학습 안 함)
  - 시작점(t=0)은 환경에서 읽은 실제 위치/속도 (attention 출력 아님)
  - cubic spline이 6개 매듭(시작 + 5)을 잇고, 미분으로 속도까지 제공
  - 컨트롤러(고정 게인 Kh,Kv,Ki)가 spline 위 한 지점을 추종

신경망 몸통(embed -> attention -> FFN -> head)은 이전과 동일,
head의 마지막 출력 차원만 3 -> 10 으로 확장.

학습: 모방학습(BC) + DAgger. 교사(APF)가 "5개 매듭 위치"를 시범으로 제공.
컨트롤러 게인은 고정(학습 안 함).

같은 폴더 필요: nav_sim.py, attn_policy.py(build_tokens 재사용), path3_spline.py
실행:  py -3.11 path3_train.py
"""

import numpy as np
import torch
import torch.nn as nn

import nav_sim as N
from attn_policy import build_tokens
from path3_spline import plan_spline_2d

# ---------------------------------------------------------------------------
# 설정 (하이퍼파라미터)
# ---------------------------------------------------------------------------
LR = 1e-4
COLLECT = 200          # 초기 교사 시범 에피소드 수
EPOCHS = 100           # BC 학습 에폭
DAGGER_ROUNDS = 5
DAGGER_EP = 80         # DAgger 라운드당 롤아웃 에피소드 수
DAG_EPOCHS = 60        # DAgger 라운드당 재학습 에폭
CAP = 400              # 에피소드 최대 스텝

# 경로(매듭) 설계
KNOT_TS_FUTURE = np.array([0.2, 0.4, 0.6, 0.8, 1.0])  # attention이 채우는 시점들
N_KNOTS = len(KNOT_TS_FUTURE)                          # 5
KNOT_TS_FULL = np.concatenate([[0.0], KNOT_TS_FUTURE]) # 시작점 포함 [0,0.2,...,1.0]
OUT_DIM = N_KNOTS * 2                                  # head 출력 = 10

AIM_RANGE = 12.0       # 각 매듭점 좌표 스케일 (ego 기준 상대, tanh*range)
PLAN_DT = 0.3          # spline에서 읽는 "앞 지점" 시점
LOOKAHEAD = 6.0        # 교사가 매듭점을 찍을 때 쓰는 진행 거리 척도
INFLUENCE = 8.0        # 교사 척력 작용 반경
D_STAR_EXPERT = 2.0    # 교사 장애물 안전여유 (감속 계산용)

# 컨트롤러 게인 (고정, 학습 안 함)
Kh = 2.2
Kv = 1.2
Ki = 0.05

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# 정책 (몸통은 이전 단계와 동일, head 출력만 10개)
# ---------------------------------------------------------------------------
class PathPolicy(nn.Module):
    def __init__(self, feat=8, d=64, heads=4, out_dim=OUT_DIM):
        super().__init__()
        self.embed = nn.Linear(feat, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, d))
        self.norm2 = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, tok):
        # tok: [B, T, feat]
        x = self.embed(tok)
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        ego = x[:, 0]                          # ego 토큰
        raw = self.head(ego)                   # [B, 10]
        pts = torch.tanh(raw) * AIM_RANGE       # 5개 점 (x,y) 상대좌표
        return pts                             # [B, 10]


# ---------------------------------------------------------------------------
# 교사(expert): APF로 진행방향 잡고, 그 방향을 따라 5개 매듭점을 배치.
#               장애물/목적지 근처면 매듭 간격을 줄여 "감속" 시범을 만든다.
# ---------------------------------------------------------------------------
def expert_path(obs):
    """반환: 길이 10 배열 [x1,y1,...,x5,y5] (ego 기준 상대 매듭점)."""
    ego = obs["ego"]
    v = ego[3]
    gx, gy = obs["goal_rel"]
    dg = np.hypot(gx, gy) + 1e-9

    # --- APF: 목적지 인력 + 장애물 척력 (0.6초 뒤 예측위치 기준) ---
    fx, fy = gx / dg, gy / dg
    min_clear = 1e9
    for dx, dy, r, vx, vy in obs["obstacles_rel"]:
        t = 0.6
        pxo, pyo = dx + vx * t, dy + vy * t
        d = np.hypot(pxo, pyo) + 1e-9
        edge = d - r - N.CAR_RADIUS
        min_clear = min(min_clear, edge)
        if edge < INFLUENCE:
            mag = (INFLUENCE - edge) / INFLUENCE
            fx -= mag * pxo / d
            fy -= mag * pyo / d
    fn = np.hypot(fx, fy) + 1e-9
    ux, uy = fx / fn, fy / fn   # 진행 단위방향

    # --- 목표속도 비율 (감속 시범): 장애물/목적지 가까우면 줄임 ---
    speed_obs = np.clip((min_clear - D_STAR_EXPERT) / 8.0, 0.15, 1.0)
    speed_goal = np.clip(dg / 8.0, 0.15, 1.0)
    vscal = min(speed_obs, speed_goal)          # 0~1

    # --- 5개 매듭점 배치 ---
    # 각 매듭의 진행거리 = (진행방향) * (속도비율 반영한 스텝 거리)
    # 느려야 하는 상황이면 매듭 간격이 좁아져 자연스러운 감속 곡선이 됨.
    step = LOOKAHEAD * vscal / N_KNOTS          # 매듭 간 진행거리
    pts = []
    cx, cy = 0.0, 0.0
    for _ in range(N_KNOTS):
        cx += ux * step
        cy += uy * step
        pts.append(cx)
        pts.append(cy)
    return np.array(pts, dtype=np.float32)


# ---------------------------------------------------------------------------
# 컨트롤러: spline 위 PLAN_DT 지점을 추종 (게인 고정)
# ---------------------------------------------------------------------------
class Controller:
    def __init__(self):
        self.i = 0.0   # 속도오차 적분항

    def reset(self):
        self.i = 0.0

    def __call__(self, obs, pts10):
        """pts10: 길이 10 (5개 점 상대좌표) -> [steer, accel]."""
        ego = obs["ego"]
        h = ego[2]
        v = ego[3]

        # world-상대 좌표계: 시작점 + 5개 매듭 (시간축 KNOT_TS_FULL)
        px = np.zeros(N_KNOTS + 1)
        py = np.zeros(N_KNOTS + 1)
        # 시작점 (t=0) = 현재 위치(원점) 기준
        px[0], py[0] = 0.0, 0.0
        for k in range(N_KNOTS):
            px[k + 1] = pts10[2 * k]
            py[k + 1] = pts10[2 * k + 1]

        # spline에서 PLAN_DT 시점의 위치/속도
        (sx, sy), (svx, svy) = plan_spline_2d(KNOT_TS_FULL, px, py, PLAN_DT)

        # 조향: spline 속도방향 추종
        theta = np.arctan2(svy, svx)
        herr = (theta - h + np.pi) % (2 * np.pi) - np.pi
        steer = float(np.clip(Kh * herr, -N.STEER_MAX, N.STEER_MAX))

        # 가속: spline 속도크기 추종 (PI)
        dv = np.hypot(svx, svy)
        ve = dv - v
        self.i += ve * N.DT
        self.i = float(np.clip(self.i, -5.0, 5.0))
        accel = float(np.clip(Kv * ve + Ki * self.i, -N.ACCEL_MAX, N.ACCEL_MAX))
        return [steer, accel]


# ---------------------------------------------------------------------------
# 롤아웃: 정책(또는 교사)으로 한 에피소드 진행. rec=True면 (토큰, 교사라벨) 수집.
# ---------------------------------------------------------------------------
def rollout(action_fn, rec=False, ctrl=None):
    env = N.NavEnv()
    obs = env.reset()
    if ctrl is None:
        ctrl = Controller()
    ctrl.reset()
    xs, ys = [], []
    done = False
    info = {"reached": False}
    steps = 0
    while not done and steps < CAP:
        pts10 = action_fn(obs)                  # 길이 10 매듭점
        if rec:
            xs.append(build_tokens(obs))
            ys.append(expert_path(obs))          # 교사 정답(라벨)
        act = ctrl(obs, pts10)
        obs, done, info = env.step(act)
        steps += 1
    return xs, ys, info


# ---------------------------------------------------------------------------
# 정책을 action_fn 형태로 감싸기
# ---------------------------------------------------------------------------
def make_policy_fn(policy):
    def fn(obs):
        tok = torch.tensor(build_tokens(obs), dtype=torch.float32,
                           device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            pts = policy(tok)[0].cpu().numpy()
        return pts
    return fn


# ---------------------------------------------------------------------------
# 학습 루프 (지도학습: 토큰 -> 교사 매듭점, MSE)
# ---------------------------------------------------------------------------
def train_on(policy, X, Y, epochs, lr=LR):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.tensor(np.array(X), dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(np.array(Y), dtype=torch.float32, device=DEVICE)
    n = len(Xt)
    bs = 256
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = policy(Xt[idx])
            loss = lossf(pred, Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# 성공률 평가
# ---------------------------------------------------------------------------
def success_rate(policy_fn, ep=40):
    s = 0
    for _ in range(ep):
        _, _, info = rollout(policy_fn)
        s += int(info["reached"])
    return s / ep


# ---------------------------------------------------------------------------
# GIF 렌더 (학습된 정책 결과)
# ---------------------------------------------------------------------------
def render_gif(policy, path="path3_result.gif"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    env = N.NavEnv()
    obs = env.reset()
    ctrl = Controller()
    policy_fn = make_policy_fn(policy)

    frames_ego = []
    frames_obs = []
    frames_aim = []     # spline에서 읽은 추종 지점 (시각화용)
    goal = np.array(N.GOAL)
    done = False
    steps = 0
    while not done and steps < CAP:
        pts10 = policy_fn(obs)
        # 시각화용: 첫 매듭점(상대) -> 절대좌표
        ex, ey, eh, ev = obs["ego"]
        aim_abs = (ex + pts10[0], ey + pts10[1])
        frames_ego.append((ex, ey, eh))
        frames_obs.append(env.obstacles.copy())
        frames_aim.append(aim_abs)
        act = ctrl(obs, pts10)
        obs, done, info = env.step(act)
        steps += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    writer = PillowWriter(fps=15)
    with writer.saving(fig, path, dpi=80):
        for t in range(len(frames_ego)):
            ax.clear()
            ax.set_xlim(0, N.WORLD_SIZE)
            ax.set_ylim(0, N.WORLD_SIZE)
            ax.set_aspect("equal")
            ax.set_title(f"step {t}  (cubic-spline path)")
            # 목적지
            ax.plot(goal[0], goal[1], "*", color="gold", markersize=20)
            # 장애물
            for ox, oy, r, vx, vy in frames_obs[t]:
                ax.add_patch(plt.Circle((ox, oy), r, color="red", alpha=0.5))
            # ego
            ex, ey, eh = frames_ego[t]
            ax.add_patch(plt.Circle((ex, ey), N.CAR_RADIUS, color="blue"))
            ax.plot([ex, ex + 2 * np.cos(eh)], [ey, ey + 2 * np.sin(eh)], "b-")
            # 추종 지점 (첫 매듭)
            ax.plot(frames_aim[t][0], frames_aim[t][1], "x",
                    color="purple", markersize=10)
        # (loop end)
    plt.close(fig)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    policy = PathPolicy().to(DEVICE)

    # 1) 교사 시범 수집 (APF 단독)
    print("collecting teacher demos...", flush=True)
    teacher_fn = expert_path
    X, Y = [], []
    t_succ = 0
    for _ in range(COLLECT):
        xs, ys, info = rollout(teacher_fn, rec=True)
        X += xs
        Y += ys
        t_succ += int(info["reached"])
    print(f"teacher success {t_succ / COLLECT * 100:.0f}% | data {len(X)}",
          flush=True)

    # 2) 행동복제 (BC)
    train_on(policy, X, Y, EPOCHS)
    policy_fn = make_policy_fn(policy)
    print(f"BC done | success {success_rate(policy_fn) * 100:.0f}%", flush=True)

    # 3) DAgger
    for rnd in range(DAGGER_ROUNDS):
        nx, ny = [], []
        for _ in range(DAGGER_EP):
            xs, ys, _ = rollout(policy_fn, rec=True)  # 정책이 방문한 상태
            nx += xs
            ny += ys
        X += nx
        Y += ny
        train_on(policy, X, Y, DAG_EPOCHS)
        policy_fn = make_policy_fn(policy)
        print(f"DAgger {rnd + 1} | success {success_rate(policy_fn) * 100:.0f}%"
              f" | data {len(X)}", flush=True)

    # 4) 최종 평가 + 저장 + GIF
    final = success_rate(policy_fn, ep=500)
    print(f"FINAL success {final * 100:.0f}%", flush=True)
    torch.save(policy.state_dict(), "path3_policy.pt")
    print("saved path3_policy.pt", flush=True)
    render_gif(policy)


if __name__ == "__main__":
    main()
