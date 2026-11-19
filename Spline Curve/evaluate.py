import torch
import path3_train as T

policy = T.PathPolicy().to(T.DEVICE)
policy.load_state_dict(torch.load("path3_policy.pt", map_location="cpu"))
policy.eval()                          # ← 이거 빠짐
policy_fn = T.make_policy_fn(policy)   # ← 이거 빠짐
final = T.success_rate(policy_fn, ep=500)
print(f"최종 성공률 {final * 100:.0f}%")
