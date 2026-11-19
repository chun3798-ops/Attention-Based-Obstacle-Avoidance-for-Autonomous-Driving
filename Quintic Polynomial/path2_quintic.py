"""5차 다항식 경계조건 풀이 (위치,속도,가속도 6개 조건 -> 계수 6개)"""
import numpy as np

def quintic_coeffs(p0, v0, a0, p1, v1, a1, T):
    """1축(스칼라) 5차 다항식 계수. x(t)=c0+c1 t+...+c5 t^5"""
    c0, c1, c2 = p0, v0, a0/2.0
    T2,T3,T4,T5 = T**2,T**3,T**4,T**5
    A = np.array([[T3,   T4,    T5   ],
                  [3*T2, 4*T3,  5*T4 ],
                  [6*T,  12*T2, 20*T3]])
    b = np.array([p1 - (c0+c1*T+c2*T2),
                  v1 - (c1+2*c2*T),
                  a1 - 2*c2])
    c3,c4,c5 = np.linalg.solve(A, b)
    return np.array([c0,c1,c2,c3,c4,c5])

def quintic_eval(coeffs, t):
    """위치와 속도를 t에서 평가"""
    c0,c1,c2,c3,c4,c5 = coeffs
    pos = c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
    vel = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
    return pos, vel

def plan_2d(p0, v0, a0, p1, v1, a1, T, t_eval):
    """2D(x,y) 한번에. p0,v0,a0,p1,v1,a1 = (x,y) 튜플. t_eval에서 위치/속도 반환"""
    cx = quintic_coeffs(p0[0],v0[0],a0[0],p1[0],v1[0],a1[0],T)
    cy = quintic_coeffs(p0[1],v0[1],a0[1],p1[1],v1[1],a1[1],T)
    px,vx = quintic_eval(cx, t_eval)
    py,vy = quintic_eval(cy, t_eval)
    return (px,py), (vx,vy)
