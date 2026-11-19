"""
Self-Attention 네비게이션 정책 (tanh head)
====================================================
구조: obs -> 토큰[ego, goal, 장애물...] -> 임베딩 -> self-attention(+FFN)
      -> ego 토큰 -> head -> tanh -> ×물리범위 -> [조향, 가속]

토큰 특징(8차원): [rel_x, rel_y, r, vx, vy, type, goal_dx, goal_dy]
  - 모든 토큰이 '목적지 방향'을 직접 알고 있어서 attention은 장애물 판단에 집중.

필요: torch, numpy, nav_sim.py
"""
import numpy as np
import torch
import torch.nn as nn
import nav_sim as N

D = 64
FEAT = 8           # rel_x, rel_y, r, vx, vy, type, goal_dx, goal_dy
HEADS = 4


def build_tokens(obs):
    x, y, h, v = obs["ego"]
    ego_vx, ego_vy = v * np.cos(h), v * np.sin(h)
    gx, gy = obs["goal_rel"]
    gxn, gyn = gx / N.WORLD_SIZE, gy / N.WORLD_SIZE     # 목적지 방향(정규화)

    toks = []
    toks.append([0.0, 0.0, N.CAR_RADIUS, ego_vx, ego_vy, 0.0, gxn, gyn])   # ego
    toks.append([gx, gy, 0.0, 0.0, 0.0, 1.0, gxn, gyn])                    # goal
    for dx, dy, r, vx, vy in obs["obstacles_rel"]:                          # 장애물
        toks.append([dx, dy, r, vx, vy, 2.0, gxn, gyn])

    t = np.array(toks, dtype=np.float32)
    t[:, 0] /= N.WORLD_SIZE     # rel_x
    t[:, 1] /= N.WORLD_SIZE     # rel_y
    t[:, 2] /= 5.0             # radius
    t[:, 3] /= N.V_MAX         # vx
    t[:, 4] /= N.V_MAX         # vy
    # 5(type), 6,7(goal) 은 그대로
    return t


class NavPolicy(nn.Module):
    def __init__(self, feat=FEAT, d=D, heads=HEADS):
        super().__init__()
        self.embed = nn.Linear(feat, d)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, d))
        self.norm2 = nn.LayerNorm(d)
        self.head  = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, tokens):
        x = self.embed(tokens)                    # [B,T,d]
        a, w = self.attn(x, x, x)                 # self-attention
        x = self.norm1(x + a)                     # 잔차 + 정규화
        x = self.norm2(x + self.ff(x))            # FFN 블록 (용량↑)
        ego = x[:, 0]                             # ego 토큰 읽기
        raw = self.head(ego)                      # [B,2]
        steer = torch.tanh(raw[:, 0]) * N.STEER_MAX
        accel = torch.tanh(raw[:, 1]) * N.ACCEL_MAX
        return torch.stack([steer, accel], dim=-1), w


policy = NavPolicy()
policy.eval()

def attn_control(obs):
    t = torch.from_numpy(build_tokens(obs)).unsqueeze(0)
    with torch.no_grad():
        a, _ = policy(t)
    return a.squeeze(0).numpy()


def load_trained(path="policy.pt"):
    """학습된 가중치 불러오기"""
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()


if __name__ == "__main__":
    import os
    if os.path.exists("policy.pt"):
        load_trained()                # 학습된 가중치 있으면 사용
        print("학습된 정책 로드됨 (policy.pt)")
    else:
        print("※ policy.pt 없음 → 랜덤 가중치 (먼저 train.py 실행)")
    N.my_algorithm = attn_control
    N.run(animate=True)
