import torch
import path2_train as T
import nav_sim as N

policy = T.PointPolicy().to("cpu")
policy.load_state_dict(torch.load("path2_policy.pt", map_location="cpu"))
policy.eval()

def make_policy_fn(policy):
    def fn(obs):
        tok = torch.tensor(T.build_tokens(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = policy(tok)[0].cpu().numpy()
        aim = out[:2]      # 목표점
        vscal = out[2]     # 목표속도
        return aim, vscal
    return fn

def success_rate(policy_fn, ep=200):
    s = 0
    for _ in range(ep):
        _, _, info = T.rollout(policy_fn)
        s += int(info["reached"])
    return s / ep

policy_fn = make_policy_fn(policy)
final = success_rate(policy_fn, ep=200)
print(f"최종 성공률 {final * 100:.0f}%")
