"""
path2_train.py로 학습한 정책의 결과를 GIF로 저장.

필요: torch, numpy, matplotlib, pillow
      + 같은 폴더에 nav_sim.py, attn_policy.py, path2_quintic.py,
        path2_train.py, path2_policy.pt
실행: py -3.11 path2_save_gif.py
결과: path2_result.gif 생성  (보라 x=목표점, 제목에 v_scale=목표속도비율 표시)
"""
import os
import torch
import path2_train as P

if os.path.exists("path2_policy.pt"):
    P.policy.load_state_dict(torch.load("path2_policy.pt", map_location="cpu"))
    P.policy.eval()
    print("학습된 정책 로드됨 (path2_policy.pt)")
else:
    print("※ path2_policy.pt 없음 → 랜덤 가중치로 렌더 (먼저 path2_train.py 실행 권장)")

P.render_gif("path2_result.gif")
