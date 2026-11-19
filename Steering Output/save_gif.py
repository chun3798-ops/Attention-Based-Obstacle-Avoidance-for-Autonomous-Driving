"""
정책을 실행해서 결과를 GIF로 저장 (창 없이 파일로).
policy.pt 가 있으면 '학습된 정책', 없으면 '랜덤 정책'으로 렌더.

필요: torch, numpy, matplotlib, pillow + nav_sim.py, attn_policy.py
실행: py -3.11 save_gif.py   → attn_result.gif 생성
"""
import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import nav_sim as N
import attn_policy as P

# 학습된 가중치가 있으면 로드
if os.path.exists("policy.pt"):
    P.load_trained("policy.pt")
    tag = "trained"
    print("학습된 정책 로드됨 (policy.pt)")
else:
    tag = "untrained(random)"
    print("policy.pt 없음 -> 랜덤 정책으로 렌더")

N.my_algorithm = P.attn_control
env = N.NavEnv(); obs = env._obs()
ego  = [env.state.copy()]
obss = [env.obstacles.copy()]
done, info = False, {}
while not done:
    obs, done, info = env.step(P.attn_control(obs))
    ego.append(env.state.copy()); obss.append(env.obstacles.copy())
ego = np.array(ego)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, N.WORLD_SIZE); ax.set_ylim(0, N.WORLD_SIZE)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
patches = [Circle((o[0], o[1]), o[2], color="#d9534f", alpha=0.6) for o in obss[0]]
for c in patches:
    ax.add_patch(c)
ax.plot(*N.GOAL,  "*", color="gold",  markersize=22, markeredgecolor="k")
ax.plot(*N.START, "o", color="green", markersize=8)
(trail,) = ax.plot([], [], "-", color="#0275d8", lw=1.8)
(car,)   = ax.plot([], [], "^", color="#0275d8", markersize=13)
title = ax.set_title("")

def upd(i):
    for c, o in zip(patches, obss[i]):
        c.center = (o[0], o[1])
    trail.set_data(ego[:i + 1, 0], ego[:i + 1, 1])
    car.set_data([ego[i, 0]], [ego[i, 1]])
    if i == len(ego) - 1:
        m = "COLLISION" if info.get("collided") else ("REACHED" if info.get("reached") else "TIMEOUT")
        title.set_text(f"{m}  ({tag})")
    else:
        title.set_text(f"step {i}  ({tag})")
    return (*patches, trail, car, title)

ani = FuncAnimation(fig, upd, frames=len(ego), interval=50, blit=False)
ani.save("attn_result.gif", writer=PillowWriter(fps=20))
print("저장 완료: attn_result.gif")
