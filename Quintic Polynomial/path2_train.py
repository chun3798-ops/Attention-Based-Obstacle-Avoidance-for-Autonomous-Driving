"""
path2 — 5차 다항식 경로 + attention이 목표점·목표속도 출력 (감속 학습)
================================================================
[매 스텝]
  현재상태(위치,속도) 읽기  ← 환경이 준 실제값 (수정 안 함)
        ↓
  attention: 토큰 보고 → 목표점(dx,dy) + 목표속도(v_target)  [학습됨]
        ↓
  5차 다항식: 시작(현재) ~ 끝(목표점,목표속도) 부드러운 곡선
        ↓
  곡선의 살짝 앞 지점(방향·속도) 추출
        ↓
  컨트롤러(고정 게인 Kh,Kv,Ki): 그 지점 추종 → [조향,가속]

- 교사 d* = 2.0 (장애물 안전여유, 고정)
- 장애물 거리에 따른 감속은 attention의 속도출력(v_scale)이 학습으로 흡수
- 목적지: 단순 거리비례 감속 (d* 없음, 도착이 목표라서)
- 장애물: 교사 d* 반영 감속

실행: py -3.11 path2_train.py  → path2_policy.pt + path2_result.gif
"""
import os, numpy as np, torch, torch.nn as nn
import nav_sim as N
from attn_policy import build_tokens
from path2_quintic import plan_2d

# ===== 고정 컨트롤러 게인 =====
Kh = 1.6; Kv = 0.8; Ki = 0.01

# ===== 교사 d* (장애물 안전여유, 고정) =====

# ===== 경로/학습 설정 =====
AIM_RANGE = 12.0          # 목표점 출력 범위(±)
PLAN_T    = 1.0           # 다항식 계획 시간지평(초)
PLAN_DT   = 0.3           # 곡선에서 '살짝 앞' 지점 시간
LOOKAHEAD = 6.0           # 교사가 목표점을 앞에 잡는 거리
INFLUENCE = 8.0           # 교사 회피 반응 반경
LR = 1e-4; COLLECT = 200; EPOCHS = 100
DAGGER_ROUNDS = 5; DAGGER_EP = 80; DAG_EPOCHS = 60
BATCH = 128; CAP = 400
EXPECT = 2 + N.N_OBSTACLES


# ---------- attention 정책: 목표점 + 목표속도 출력 ----------
class PointPolicy(nn.Module):
    def __init__(self, feat=8, d=64, heads=4):
        super().__init__()
        self.embed = nn.Linear(feat, d)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Linear(128,d))
        self.norm2 = nn.LayerNorm(d)
        self.head  = nn.Sequential(nn.Linear(d,128), nn.ReLU(),
                                   nn.Linear(128,64), nn.ReLU(), nn.Linear(64,3))
    def forward(self, tok):
        x = self.embed(tok); a,_ = self.attn(x,x,x)
        x = self.norm1(x+a); x = self.norm2(x+self.ff(x))
        raw = self.head(x[:,0])
        aim   = torch.tanh(raw[:, :2]) * AIM_RANGE       # 목표점(상대 dx,dy)
        vscal = torch.sigmoid(raw[:, 2])                 # 목표속도 비율 0~1
        return aim, vscal


# ---------- 컨트롤러: 5차 다항식 계획 후 추종 (게인 고정) ----------
class Controller:
    def __init__(self): self.i = 0.0
    def reset(self): self.i = 0.0
    def __call__(self, obs, aim, vscal):
        x, y, h, v = obs["ego"]
        # world축(원점=ego)으로 통일. 시작속도는 현재 헤딩 방향으로.
        p0=(0.0,0.0); v0=(v*np.cos(h), v*np.sin(h)); a0=(0.0,0.0)
        dx,dy=aim; dn=np.hypot(dx,dy)+1e-6
        v_target=N.V_MAX*float(vscal)
        p1=(dx,dy); v1=(v_target*dx/dn, v_target*dy/dn); a1=(0.0,0.0)
        # 다항식으로 살짝 앞 지점의 위치·속도 (world축)
        (px,py),(pvx,pvy)=plan_2d(p0,v0,a0,p1,v1,a1, T=PLAN_T, t_eval=PLAN_DT)
        # 그 지점으로의 방향(world) → 헤딩오차
        theta=np.arctan2(py,px)
        herr=np.arctan2(np.sin(theta-h), np.cos(theta-h))
        steer=np.clip(Kh*herr, -N.STEER_MAX, N.STEER_MAX)
        # 곡선이 알려주는 그 지점의 속도크기를 추종
        dv=np.hypot(pvx,pvy); ve=dv-v; self.i+=ve*N.DT
        accel=np.clip(Kv*ve+Ki*self.i, -N.ACCEL_MAX, N.ACCEL_MAX)
        return np.array([steer,accel])

ctrl=Controller(); policy=PointPolicy()
opt=torch.optim.Adam(policy.parameters(), lr=LR); lossf=nn.MSELoss()


# ---------- 교사: 목표점(APF) + 목표속도(거리기반 감속, 장애물 d*) ----------
def expert(obs):
    x,y,h,v=obs["ego"]; gx,gy=obs["goal_rel"]
    dg=np.hypot(gx,gy)+1e-6; fx,fy=gx/dg, gy/dg
    min_clear=1e9
    for dx,dy,r,vx,vy in obs["obstacles_rel"]:
        t=0.6; pxo,pyo=dx+vx*t, dy+vy*t; d=np.hypot(pxo,pyo)+1e-6
        edge=d-r-N.CAR_RADIUS                       # 표면까지 여유
        min_clear=min(min_clear, edge)
        if edge<INFLUENCE:
            mag=max(1.0/max(edge,0.5)-1.0/INFLUENCE, 0)*4.0; fx-=mag*pxo/d; fy-=mag*pyo/d
    n=np.hypot(fx,fy)+1e-6
    aim=np.array([fx/n*LOOKAHEAD, fy/n*LOOKAHEAD], dtype=np.float32)
    # 목표속도: 장애물(d* 반영)과 목적지(거리, d* 없음) 중 더 위험한 쪽
    speed_obs  = np.clip(min_clear/8.0, 0.15, 1.0)                     # 장애물 (d* 없음, 표면까지 실거리)
    speed_goal = np.clip(dg/8.0, 0.15, 1.0)                            # 목적지 단순거리
    vscal=float(min(speed_obs, speed_goal))
    return aim, vscal

def policy_out(obs):
    with torch.no_grad():
        aim,vscal=policy(torch.tensor(build_tokens(obs)).unsqueeze(0))
    return aim.squeeze(0).numpy(), float(vscal)


def rollout(out_fn, rec=False, cap=CAP):
    ctrl.reset(); env=N.NavEnv(); obs=env._obs(); done=False; info={}; X,Y=[],[]; k=0
    while not done and k<cap:
        aim,vscal=out_fn(obs)
        if rec:
            tok=build_tokens(obs)
            if tok.shape[0]==EXPECT:
                eaim,evs=expert(obs); X.append(tok); Y.append([eaim[0],eaim[1],evs])
        obs,done,info=env.step(ctrl(obs,aim,vscal)); k+=1
    return X,Y,info

def make_Y(Ylist):
    Y=np.array(Ylist,dtype=np.float32); Y[:, :2]/=AIM_RANGE   # 목표점만 정규화, 속도는 0~1 그대로
    return torch.tensor(Y)

def train_on(X,Y,epochs):
    policy.train(); n=len(X); L=0
    for e in range(epochs):
        perm=torch.randperm(n)
        for i in range(0,n,BATCH):
            idx=perm[i:i+BATCH]
            aim,vscal=policy(X[idx])
            pred=torch.cat([aim/AIM_RANGE, vscal.unsqueeze(1)], dim=1)
            loss=lossf(pred, Y[idx]); opt.zero_grad(); loss.backward(); opt.step(); L=loss.item()
    return L

def succ(ep=40):
    s=0
    for _ in range(ep):
        _,_,info=rollout(policy_out); s+=int(info["reached"])
    return s/ep


def render_gif(path="path2_result.gif"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle
    ctrl.reset(); env=N.NavEnv(); obs=env._obs()
    ego=[env.state.copy()]; obss=[env.obstacles.copy()]; vs=[0.0]
    a0,vs0=policy_out(obs); tgt=[a0+obs["ego"][:2]]
    done=False; info={}; k=0
    while not done and k<CAP:
        aim,vscal=policy_out(obs); tgt.append(aim+obs["ego"][:2]); vs.append(vscal)
        obs,done,info=env.step(ctrl(obs,aim,vscal)); k+=1
        ego.append(env.state.copy()); obss.append(env.obstacles.copy())
    ego=np.array(ego)
    fig,ax=plt.subplots(figsize=(6,6)); ax.set_xlim(0,N.WORLD_SIZE); ax.set_ylim(0,N.WORLD_SIZE)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    patches=[Circle((o[0],o[1]),o[2],color="#d9534f",alpha=0.6) for o in obss[0]]
    for c in patches: ax.add_patch(c)
    ax.plot(*N.GOAL,"*",color="gold",markersize=22,markeredgecolor="k"); ax.plot(*N.START,"o",color="green",markersize=8)
    (trail,)=ax.plot([],[], "-",color="#0275d8",lw=1.8)
    (car,)=ax.plot([],[], "^",color="#0275d8",markersize=13)
    (aimp,)=ax.plot([],[], "x",color="purple",markersize=10)
    title=ax.set_title("")
    def upd(i):
        for c,o in zip(patches,obss[i]): c.center=(o[0],o[1])
        trail.set_data(ego[:i+1,0],ego[:i+1,1]); car.set_data([ego[i,0]],[ego[i,1]])
        ti=min(i,len(tgt)-1); aimp.set_data([tgt[ti][0]],[tgt[ti][1]])
        vi=min(i,len(vs)-1)
        if i==len(ego)-1:
            m="REACHED" if info.get("reached") else ("COLLISION" if info.get("collided") else "TIMEOUT")
            title.set_text(f"{m}  v_scale={vs[vi]:.2f}")
        else: title.set_text(f"step {i}  v_scale={vs[vi]:.2f}  (x=aim)")
        return (*patches,trail,car,aimp,title)
    ani=FuncAnimation(fig,upd,frames=len(ego),interval=50,blit=False)
    ani.save(path,writer=PillowWriter(fps=20)); print("저장:",path,flush=True)


if __name__=="__main__":
    print(f"게인(고정) Kh={Kh} Kv={Kv} Ki={Ki}", flush=True)
    X,Y,s=[],[],0
    for _ in range(COLLECT):
        xs,ys,info=rollout(expert, rec=True); X+=xs; Y+=ys; s+=int(info["reached"])
    print(f"교사 성공률 {s/COLLECT*100:.0f}% | 데이터 {len(X)}", flush=True)
    X=torch.tensor(np.array(X,dtype=np.float32)); Y=make_Y(Y)

    train_on(X,Y,EPOCHS); policy.eval()
    print(f"BC 후 성공률 {succ(40)*100:.0f}%", flush=True)
    for rnd in range(DAGGER_ROUNDS):
        nx,ny=[],[]
        for _ in range(DAGGER_EP):
            xs,ys,_=rollout(policy_out, rec=True); nx+=xs; ny+=ys
        X=torch.cat([X,torch.tensor(np.array(nx,dtype=np.float32))]); Y=torch.cat([Y,make_Y(ny)])
        train_on(X,Y,DAG_EPOCHS); policy.eval()
        print(f"DAgger {rnd+1} 후 성공률 {succ(40)*100:.0f}% | 데이터 {len(X)}", flush=True)

    torch.save(policy.state_dict(),"path2_policy.pt")
    print(f"최종 성공률 {succ(500)*100:.0f}%", flush=True)
    render_gif()
    print("DONE", flush=True)
