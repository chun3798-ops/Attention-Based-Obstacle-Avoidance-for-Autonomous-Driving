"""
attn_policy 학습 (모방학습 + 가벼운 DAgger) — PPO 없음
- 가중치 초기값: PyTorch 기본 (자동) / 학습률: 1e-4
- 교사: 인공 포텐셜 필드(APF, 부드러운 속도제어)
- DAgger: 정책이 가본 상태에 교사 정답을 덧붙여 재학습 (분포이동 완화)
실행: py -3.11 train.py   (결과: policy.pt)
"""
import numpy as np, torch, torch.nn as nn
import nav_sim as N
from attn_policy import NavPolicy, build_tokens

LR=1e-4; COLLECT=180; DAGGER_ROUNDS=5; DAGGER_EP=80
EPOCHS=120; DAG_EPOCHS=60; BATCH=128; INFLUENCE=8.0; CAP=300
EXP_TOK=2+N.N_OBSTACLES
LIMITS=np.array([N.STEER_MAX,N.ACCEL_MAX],dtype=np.float32); LIMITS_T=torch.tensor(LIMITS)

def expert_action(obs):
    x,y,h,v=obs["ego"]; gx,gy=obs["goal_rel"]
    dg=np.hypot(gx,gy)+1e-6; fx,fy=gx/dg,gy/dg; clr=INFLUENCE
    for dx,dy,r,vx,vy in obs["obstacles_rel"]:
        t=0.6; px,py=dx+vx*t,dy+vy*t; d=np.hypot(px,py)+1e-6
        edge=d-r-N.CAR_RADIUS; clr=min(clr,np.hypot(dx,dy)-r-N.CAR_RADIUS)
        if edge<INFLUENCE:
            mag=max(1.0/max(edge,0.5)-1.0/INFLUENCE,0)*4.0; fx-=mag*px/d; fy-=mag*py/d
    target=np.arctan2(fy,fx); err=np.arctan2(np.sin(target-h),np.cos(target-h))
    steer=np.clip(2.5*err,-N.STEER_MAX,N.STEER_MAX)
    align=max(np.cos(err),0.0); clr=max(clr,0.0)
    dv=N.V_MAX*align*np.clip(clr/INFLUENCE,0.25,1.0)
    accel=np.clip(2.0*(dv-v),-N.ACCEL_MAX,N.ACCEL_MAX)
    return np.array([steer,accel],dtype=np.float32)

def rollout(control,rec=False,cap=CAP):
    env=N.NavEnv(); obs=env._obs(); done=False; info={}; X,Y=[],[]; k=0
    while not done and k<cap:
        if rec:
            tok=build_tokens(obs)
            if tok.shape[0]==EXP_TOK: X.append(tok); Y.append(expert_action(obs))
        obs,done,info=env.step(control(obs)); k+=1
    return X,Y,info

policy=NavPolicy(); opt=torch.optim.Adam(policy.parameters(),lr=LR); lossf=nn.MSELoss()
def pact(obs):
    with torch.no_grad():
        a,_=policy(torch.tensor(build_tokens(obs)).unsqueeze(0))
    return a.squeeze(0).numpy()
def train_on(X,Y,ep):
    policy.train(); n=len(X)
    for e in range(ep):
        perm=torch.randperm(n)
        for i in range(0,n,BATCH):
            idx=perm[i:i+BATCH]; pred,_=policy(X[idx])
            loss=lossf(pred/LIMITS_T,Y[idx]); opt.zero_grad(); loss.backward(); opt.step()
def succ(ep):
    s=0
    for _ in range(ep):
        _,_,info=rollout(pact); s+=int(info["reached"])
    return s/ep

print("교사 데이터 수집...",flush=True)
X,Y,s=[],[],0
for _ in range(COLLECT):
    xs,ys,info=rollout(expert_action,rec=True); X+=xs; Y+=ys; s+=int(info["reached"])
print(f"  교사 성공률 {s/COLLECT*100:.0f}% | 데이터 {len(X)}",flush=True)
X=torch.tensor(np.array(X,dtype=np.float32)); Y=torch.tensor(np.array(Y,dtype=np.float32)/LIMITS)

print("초기 모방학습...",flush=True); train_on(X,Y,EPOCHS)
policy.eval(); print(f"  BC 후 성공률 {succ(40)*100:.0f}%",flush=True)

for rnd in range(DAGGER_ROUNDS):
    nx,ny=[],[]
    for _ in range(DAGGER_EP):
        xs,ys,_=rollout(pact,rec=True); nx+=xs; ny+=ys
    X=torch.cat([X,torch.tensor(np.array(nx,dtype=np.float32))])
    Y=torch.cat([Y,torch.tensor(np.array(ny,dtype=np.float32)/LIMITS)])
    train_on(X,Y,DAG_EPOCHS); policy.eval()
    print(f"  DAgger {rnd+1} 후 성공률 {succ(40)*100:.0f}% | 데이터 {len(X)}",flush=True)

torch.save(policy.state_dict(),"policy.pt"); policy.eval()
print(f"최종 성공률 {succ(500)*100:.0f}% (랜덤맵 500개)",flush=True); print("DONE",flush=True)
