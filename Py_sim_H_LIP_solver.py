"""
Author: Nhan Le
Date: 2026-01-19
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.integrate import solve_ivp


def dynamic_eq(t, x, ld):
    """Single-support H-LIP dynamics: x = [CoM position; CoM velocity]."""
    return np.array([x[1], ld**2 * x[0]])

def animate_xz_com_foot(x_pos, foot_pos, z0=1.0,
                        frames_per_segment=30,
                        base_pause=0.02,
                        speedup_with_distance=True):
    """
    Animate CoM + foot in the x–z plane (1D walking visualization).

    Parameters
    ----------
    x_pos : array-like, shape (N,) or (N,1)
        CoM x positions along the trajectory.
    foot_pos : array-like, shape (N,) or (N,1)
        Foot x positions at the same nodes as x_pos.
        Foot stays fixed during each segment i->i+1, then updates at node i+1.
    z0 : float
        CoM height.
    frames_per_segment : int
        Frames to interpolate per segment.
    base_pause : float
        Base pause duration (seconds) used to control animation speed.
    speedup_with_distance : bool
        If True, larger step distance animates faster (smaller pause per frame).
    """

    # --- sanitize inputs ---
    x_pos = np.asarray(x_pos).reshape(-1)      # (N,)
    foot_pos = np.asarray(foot_pos).reshape(-1)

    if x_pos.size != foot_pos.size:
        raise ValueError("x_pos and foot_pos must have the same length.")
    if x_pos.size < 2:
        raise ValueError("Need at least 2 points to animate.")
    if z0 is None or (isinstance(z0, float) and np.isnan(z0)):
        z0 = 1.0

    N = x_pos.size

    # --- figure setup ---
    plt.ion()  # interactive on (safe if already on)
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("CoM + Foot (x–z animation)")

    allx = np.concatenate([x_pos, foot_pos])
    xmin, xmax = np.min(allx), np.max(allx)
    pad = 0.1 * max(xmax - xmin, 1e-6)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(-0.2 * z0, 1.2 * z0)

    # optional: faint guide lines
    ax.plot([xmin - pad, xmax + pad], [0, 0], "-")                 # ground
    ax.plot(x_pos, np.full(N, z0), ":")                            # CoM path
    ax.plot(foot_pos, np.zeros(N), ":")                            # foot path

    # --- graphics objects ---
    (hCom,) = ax.plot([x_pos[0]], [z0], "o", markersize=8)
    (hFoot,) = ax.plot([foot_pos[0]], [0], "s", markersize=8)
    (hLeg,) = ax.plot([foot_pos[0], x_pos[0]], [0, z0], "-", linewidth=2)

    fig.canvas.draw()
    fig.canvas.flush_events()

    # --- animation loop ---
    for i in range(N - 1):
        foot = foot_pos[i]   # fixed during this segment
        x0 = x_pos[i]
        x1 = x_pos[i + 1]

        dist = abs(x1 - x0)

        if speedup_with_distance:
            pause_per_frame = base_pause / max(dist, 0.05)  # farther => faster
            pause_per_frame = min(max(pause_per_frame, 0.002), 0.05)
        else:
            pause_per_frame = base_pause

        for k in range(frames_per_segment + 1):
            t = k / frames_per_segment
            com = (1 - t) * x0 + t * x1

            hCom.set_data([com], [z0])
            hFoot.set_data([foot], [0])
            hLeg.set_data([foot, com], [0, z0])

            fig.canvas.draw_idle()
            plt.pause(pause_per_frame)

        # update foot at node
        foot = foot_pos[i + 1]
        hFoot.set_data([foot], [0])
        hLeg.set_data([foot, x1], [0, z0])
        fig.canvas.draw_idle()
        plt.pause(0.001)

    plt.ioff()
    plt.show()


# ==============================
# Main Function
# ==============================
def main(show_plot=True):
    # Constants / Parameters
    z0 = 0.5           # m (COM height)
    g  = 9.81 

    # Controller variable 
    Ts = 0.2           # SSP
    Td = 0.1           # DSP
    xi = np.array([[-0.05],[0.5]])   # initial state
    xd = np.array([[-0.1],[1]])      # desired state
    stoptime = 2.1     # desired stop time

    # initial calculations
    ld = np.sqrt(g/z0)
    K  = np.array([[1.0, Td + (1/ld)*np.cosh(ld*Ts)/np.sinh(ld*Ts)]])  
    ud = float(xd[1,0] * (Ts + Td))  # desired step length in P1 orbit
    xf = xi
    i = 0                        # initial start time
    u = np.array([0.0])          # initial cartesian step
    
    # ---- Run ode45-equivalent solver ----
    while i <= stoptime:
        # integrate within-step dynamics
        sol = solve_ivp(
            fun=lambda t, x: dynamic_eq(t, x, ld),
            t_span=(0.0, Ts),
            y0=xf[:, -1].reshape(-1),
            method="RK45",
            max_step=1e-3,
            rtol=1e-9,
            atol=1e-12,
        )
        x = sol.y[:, -1]

        # compute foot placement input for THIS step transition
        e = xf[:, -1].reshape(2, 1) - xd
        uk = float(ud + (K @ e)[0, 0])

        u = np.append(u, u[-1] + uk)

        # state reset
        x_next = np.array([x[0] + x[1]*Td - uk, x[1]], dtype=float)
        xf = np.column_stack((xf, x_next))
        i += Ts + Td

    x_Cart = xf[0, :] + u
    print("CoM x positions:", x_Cart)
    print("Foot x positions:", u)
    if show_plot:
        animate_xz_com_foot(x_Cart, u, z0)



# ==============================
# Script Entry Point
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the H-LIP ode45-style Python solver.")
    parser.add_argument("--no-plot", action="store_true", help="print the trajectory without opening the animation")
    args = parser.parse_args()
    main(show_plot=not args.no_plot)
