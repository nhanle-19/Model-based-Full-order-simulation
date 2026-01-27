import numpy as np
import matplotlib.pyplot as plt
from math import comb

def bezier(t, t0, t1, P):
    P = np.asarray(P, dtype=float)
    n = len(P) - 1
    s = (t - t0) / (t1 - t0)
    s = np.clip(s, 0.0, 1.0)

    p = np.zeros_like(s)
    for i in range(n + 1):
        p += comb(n, i) * (1 - s)**(n - i) * s**i * P[i]
    return p

def main():
    t0, t1 = 0.0, 0.4
    t = np.linspace(t0, t1, 400)

    Px = [0.0, 0.0, 0.10, 0.25, 0.40, 0.50]  # x control points
    Pz = [0.0, 0.0, 0.12, 0.12, 0.00, 0.00]  # z control points

# -----------------------
# Evaluate Bezier
# -----------------------
    x = bezier(t, t0, t1, Px)
    z = bezier(t, t0, t1, Pz)

# -----------------------
# Plots
# -----------------------
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("time (s)")
    plt.ylabel("x(t)")
    plt.title("Bezier x(t)")
    plt.grid()

    plt.figure()
    plt.plot(t, z)
    plt.xlabel("time (s)")
    plt.ylabel("z(t)")
    plt.title("Bezier z(t)")
    plt.grid()

    plt.figure()
    plt.plot(x, z)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("Swing foot path (x-z)")
    plt.grid()

    plt.show()

if __name__ == "__main__":
    main()