"""
path3_save_gif.py
-----------------
학습된 path3_policy.pt 를 불러와서 결과 GIF만 다시 렌더.

같은 폴더 필요: nav_sim.py, attn_policy.py, path3_spline.py, path3_train.py
실행:  py -3.11 path3_save_gif.py
  -> path3_result.gif 생성
"""

import os
import torch

import path3_train as T


def main():
    policy = T.PathPolicy().to(T.DEVICE)
    if os.path.exists("path3_policy.pt"):
        policy.load_state_dict(torch.load("path3_policy.pt", map_location="cpu"))
        print("loaded path3_policy.pt", flush=True)
    else:
        print("path3_policy.pt 없음 -> 랜덤 가중치로 렌더 (구조 확인용)",
              flush=True)
    policy.eval()
    T.render_gif(policy, path="path3_result.gif")


if __name__ == "__main__":
    main()
