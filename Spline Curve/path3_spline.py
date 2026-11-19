"""
path3_spline.py
----------------
시간 고정 매듭(knot)에 대한 cubic spline 경로 생성기.

설계:
  - 시작점(t=0) + attention이 찍는 점 5개(t=0.2,0.4,0.6,0.8,1.0)
    => 총 6개 매듭을 시간축으로 잇는 부드러운 곡선 x(t), y(t)
  - x축, y축을 각각 1D natural cubic spline으로 보간
  - 위치는 spline 평가, 속도는 spline 미분으로 얻음 (별도 속도출력 불필요)

scipy 없이 직접 구현 (numpy만 사용) — 의존성/설치 문제 회피.
natural cubic spline: 양 끝 2차미분 = 0 (자연 경계조건).
"""

import numpy as np


def _solve_tridiagonal(a, b, c, d):
    """삼중대각 연립방정식 Thomas 알고리즘.
    a: 하부대각(len n, a[0] 미사용), b: 주대각(len n),
    c: 상부대각(len n, c[-1] 미사용), d: 우변(len n).
    반환: 해 x (len n).
    """
    n = len(b)
    cc = np.zeros(n)
    dd = np.zeros(n)
    cc[0] = c[0] / b[0]
    dd[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cc[i - 1]
        cc[i] = c[i] / m if i < n - 1 else 0.0
        dd[i] = (d[i] - a[i] * dd[i - 1]) / m
    x = np.zeros(n)
    x[-1] = dd[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dd[i] - cc[i] * x[i + 1]
    return x


class CubicSpline1D:
    """1D natural cubic spline. ts(매듭 시간) 오름차순, ys(값)."""

    def __init__(self, ts, ys):
        self.ts = np.asarray(ts, dtype=float)
        self.ys = np.asarray(ys, dtype=float)
        n = len(self.ts)
        self.n = n
        h = np.diff(self.ts)  # 구간 길이 (len n-1)
        self.h = h

        # 2차미분 m[i] 를 구하는 선형계 (natural: m[0]=m[n-1]=0)
        m = np.zeros(n)
        if n >= 3:
            # 내부 점들에 대한 삼중대각계 (크기 n-2)
            a = np.zeros(n - 2)  # 하부
            b = np.zeros(n - 2)  # 주대각
            c = np.zeros(n - 2)  # 상부
            d = np.zeros(n - 2)  # 우변
            for i in range(1, n - 1):
                k = i - 1
                a[k] = h[i - 1]
                b[k] = 2.0 * (h[i - 1] + h[i])
                c[k] = h[i]
                d[k] = 6.0 * ((self.ys[i + 1] - self.ys[i]) / h[i]
                              - (self.ys[i] - self.ys[i - 1]) / h[i - 1])
            m_inner = _solve_tridiagonal(a, b, c, d)
            m[1:n - 1] = m_inner
        self.m = m  # 2차미분 계수

    def eval(self, t):
        """위치 x(t). t는 스칼라."""
        ts, ys, h, m = self.ts, self.ys, self.h, self.m
        # 구간 인덱스 찾기 (clamp)
        if t <= ts[0]:
            i = 0
        elif t >= ts[-1]:
            i = self.n - 2
        else:
            i = int(np.searchsorted(ts, t) - 1)
            i = max(0, min(i, self.n - 2))
        dt = t - ts[i]
        hi = h[i]
        a = ys[i]
        b = (ys[i + 1] - ys[i]) / hi - hi * (2 * m[i] + m[i + 1]) / 6.0
        c = m[i] / 2.0
        d = (m[i + 1] - m[i]) / (6.0 * hi)
        return a + b * dt + c * dt * dt + d * dt * dt * dt

    def deriv(self, t):
        """속도 x'(t). t는 스칼라."""
        ts, ys, h, m = self.ts, self.ys, self.h, self.m
        if t <= ts[0]:
            i = 0
        elif t >= ts[-1]:
            i = self.n - 2
        else:
            i = int(np.searchsorted(ts, t) - 1)
            i = max(0, min(i, self.n - 2))
        dt = t - ts[i]
        hi = h[i]
        b = (ys[i + 1] - ys[i]) / hi - hi * (2 * m[i] + m[i + 1]) / 6.0
        c = m[i] / 2.0
        d = (m[i + 1] - m[i]) / (6.0 * hi)
        return b + 2 * c * dt + 3 * d * dt * dt


def plan_spline_2d(knot_ts, pts_x, pts_y, t_eval):
    """2D cubic spline 경로를 만들고 t_eval 시점의 위치/속도 반환.

    knot_ts : 매듭 시간 배열 (예: [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    pts_x   : 각 매듭의 x 좌표 (시작점 포함, knot_ts와 같은 길이)
    pts_y   : 각 매듭의 y 좌표
    t_eval  : 평가 시점 (스칼라)

    반환: (px, py), (vx, vy)
    """
    sx = CubicSpline1D(knot_ts, pts_x)
    sy = CubicSpline1D(knot_ts, pts_y)
    px, py = sx.eval(t_eval), sy.eval(t_eval)
    vx, vy = sx.deriv(t_eval), sy.deriv(t_eval)
    return (px, py), (vx, vy)


if __name__ == "__main__":
    # 간단 자가검증 (테스트용 — 실제 학습과 무관)
    ts = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    xs = np.array([0.0, 1.0, 2.5, 3.0, 5.0, 6.0])
    ys = np.array([0.0, 0.5, 0.3, 1.2, 0.8, 1.5])
    for te in [0.0, 0.3, 0.5, 1.0]:
        (px, py), (vx, vy) = plan_spline_2d(ts, xs, ys, te)
        print(f"t={te}: pos=({px:.3f},{py:.3f}) vel=({vx:.3f},{vy:.3f})")
    # 매듭점에서 위치가 정확히 일치하는지 확인
    (px, _), _ = plan_spline_2d(ts, xs, ys, 0.4)
    assert abs(px - 2.5) < 1e-9, "knot 위치 불일치"
    print("OK: spline passes through knots")
