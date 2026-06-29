"""
Full-order input-output linearization with H-LIP (Hybrid Linear Inverted Pendulum) stepping.

Uses a five-link planar walker: IO linearization for swing-foot and COM tracking,
with H-LIP deadbeat stepping law for step-length adaptation (level ground).

Author: Nhan Le
Date: 2026-01-19
"""

# ==============================
# Imports
# ==============================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import comb
from pathlib import Path

try:
    import pinocchio as pin
except ModuleNotFoundError as exc:
    pin = None
    PINOCCHIO_IMPORT_ERROR = exc
else:
    PINOCCHIO_IMPORT_ERROR = None

# ==============================
# Model definition
# ==============================
# Load URDF with planar root (no floating base; motion in xz plane)
URDF_PATH = Path(__file__).with_name("five_link_walker.urdf")
urdf = str(URDF_PATH)

if pin is not None and URDF_PATH.exists():
    model = pin.buildModelFromUrdf(urdf, root_joint=pin.JointModelPlanar())
else:
    model = None

# ==============================
# Functions
# ==============================
def ensure_model_available():
    """Validate optional full-order dependencies before running the simulation."""
    if not URDF_PATH.exists():
        raise FileNotFoundError(
            f"Missing robot model: {URDF_PATH.name}. Place the URDF next to this script "
            "before running the full-order examples."
        )
    if pin is None:
        raise RuntimeError(
            "Pinocchio is required for the full-order examples. "
            "Install the `pin` package, then run this script again."
        )


def joint_idx_q(model, joint_name):
    """Return the index of the joint's first coordinate in the configuration vector q."""
    jid = model.getJointId(joint_name)
    return model.joints[jid].idx_q

def set_continuous_joint_angle_in_q(model, q, joint_name, theta):
    """Store continuous joint angle theta in q as (cos(theta), sin(theta)) for the given joint."""
    i = joint_idx_q(model, joint_name)
    q[i]   = np.cos(theta)
    q[i+1] = np.sin(theta)

def joint_idx_v(model, joint_name):
    """Return the index of the joint's first coordinate in the velocity vector v."""
    jid = model.getJointId(joint_name)
    return model.joints[jid].idx_v

def get_continuous_joint_angle_from_q(model, q, joint_name):
    """Extract continuous joint angle from q (stored as cos/sin) via arctan2."""
    i = joint_idx_q(model, joint_name)
    return np.arctan2(q[i+1], q[i])

def set_joint_angle_in_q(model, q, joint_name, theta):
    """Set a revolute joint angle theta directly in the configuration vector q."""
    jid = model.getJointId(joint_name)
    if jid == 0:
        raise ValueError(f"Joint '{joint_name}' not found in URDF.")
    i = model.joints[jid].idx_q
    q[i] = theta


def add_point_frame(model, parent_frame_name, new_frame_name, xyz_local):
    """
    Add an operational frame at a point offset from a parent frame (e.g. foot point at shin end).
    Used to define swing/stance foot points for the walker.
    """
    parent_fid = model.getFrameId(parent_frame_name)
    parent_frame = model.frames[parent_fid]
    parent_joint = parent_frame.parentJoint
    X_parent_new = pin.SE3(np.eye(3), np.array(xyz_local))
    X_joint_new = parent_frame.placement * X_parent_new
    new_frame = pin.Frame(
        new_frame_name,
        parent_joint,
        parent_fid,
        X_joint_new,
        pin.FrameType.OP_FRAME
    )
    model.addFrame(new_frame)


def qdot_from_sincos(q, v):
    """
    Compute qdot when q stores continuous joints as (cos, sin): d(cos)/dt = -sin*thdot, d(sin)/dt = cos*thdot.
    Assumes 4 such joints (indices 0:2, 2:4, 4:6, 6:8) and v has corresponding angular rates.
    """
    qdot = np.zeros_like(q)
    for k in range(4):
        c = q[2*k]
        s = q[2*k + 1]
        thdot = v[k]
        qdot[2*k]     = -s * thdot
        qdot[2*k + 1] =  c * thdot
    return qdot

def build_step_bezier_points(x0, z0, xT, zT, z_peak=0.12):
    """
    6 control points (order 5). Simple shape:
      - first two points at start (zero initial slope-ish)
      - middle points push toward peak height
      - last two points at end (zero final slope-ish)
    """
    Px = np.array([x0, x0, x0 + 0.02, xT - 0.02, xT, xT], dtype=float)
    Pz = np.array([z0, z0, z_peak,    z_peak,    zT, zT], dtype=float)
    return Px, Pz

def bezier(t, t0, t1, P):
    """
    Evaluate Bezier curve and its first two time-derivatives at time t.
    P: control points; t mapped to s in [0,1] via s = (t-t0)/(t1-t0).
    Returns (position p, velocity hd, acceleration hdd) in time domain.
    """
    P = np.asarray(P, dtype=float)
    n = len(P) - 1
    s = (t - t0) / (t1 - t0)
    s = np.clip(s, 0.0, 1.0)

    # Position: sum over Bernstein terms
    p = np.zeros_like(s)
    for i in range(n + 1):
        p += comb(n, i) * (1 - s)**(n - i) * s**i * P[i]

    # First derivative w.r.t. s (then scale by 1/T for time)
    dp_ds = np.zeros_like(s)
    for i in range(n):
        dp_ds += comb(n - 1, i) * (1 - s)**(n - 1 - i) * s**i * (P[i + 1] - P[i])
    dp_ds *= n

    # Second derivative w.r.t. s (then scale by 1/T^2 for time)
    d2p_ds2 = np.zeros_like(s)
    for i in range(n - 1):
        d2p_ds2 += comb(n - 2, i) * (1 - s)**(n - 2 - i) * s**i * (P[i + 2] - 2 * P[i + 1] + P[i])
    d2p_ds2 *= n * (n - 1)

    T = (t1 - t0)
    hd = dp_ds / T
    hdd = d2p_ds2 / (T**2)

    return p, hd, hdd

def com_height_relative_to_stance_z(model, data, q, stance_fid):
    """
    Returns CoM height relative to stance foot origin (world Z direction).
    """

    # Update kinematics
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # --- CoM in world ---
    pin.centerOfMass(model, data, q)
    z_com = data.com[0][2]

    # --- stance foot origin in world ---
    z_stance = data.oMf[stance_fid].translation[2]

    # --- relative height ---
    return float(z_com - z_stance)

def com_pos_rel_stance_x(model, data, q, stance_fid):
    """
    Returns horizontal CoM position relative to stance foot
    along world X direction.
    """

    # Update kinematics
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # --- CoM world position ---
    pin.centerOfMass(model, data, q)
    x_com = data.com[0][0]   # world X

    # --- stance foot world position ---
    x_stance = data.oMf[stance_fid].translation[0]

    return float(x_com - x_stance)

def com_vel_rel_stance_x(model, data, q, v, stance_fid):
    """Horizontal (world-x) velocity of CoM relative to stance foot: v_com_x - v_stance_x."""
    pin.forwardKinematics(model, data, q, v)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)

    pin.centerOfMass(model, data, q, v)
    Jcom = pin.jacobianCenterOfMass(model, data, q)  # 3 x nv
    vcom_x = float(Jcom[0, :] @ v)

    J6_st = pin.computeFrameJacobian(model, data, q, stance_fid, pin.LOCAL_WORLD_ALIGNED)
    vstance_x = float(J6_st[0, :] @ v)

    return vcom_x - vstance_x

def compute_Jc(model, data, q, v, stance_fid):
    """Constraint Jacobian for stance foot: linear part in x and z (world-aligned). Shape (2, nv)."""
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)
    Jc = pin.computeFrameJacobian(model, data, q, stance_fid, pin.LOCAL_WORLD_ALIGNED)
    Jc_lin = Jc[[0,2], :]

    return Jc_lin

def compute_Jdotv_h(model, data, q, v, stance_fid, swing_fid, torso_fid):
    """
    J̇*q̇ for output h = (swing - stance) position in xz. With qdd=0, classical acceleration = J̇*q̇.
    Returns 2-vector: (J̇_h @ v) for x and z components of swing relative to stance.
    """
    pin.forwardKinematics(model, data, q, v, np.zeros(model.nv))
    pin.updateFramePlacements(model, data)

    a_stance = pin.getFrameClassicalAcceleration(model, data, stance_fid, pin.LOCAL_WORLD_ALIGNED)
    a_swing  = pin.getFrameClassicalAcceleration(model, data, swing_fid,  pin.LOCAL_WORLD_ALIGNED)
    Jdotv_stance_xz = a_stance.linear[[0, 2]]
    Jdotv_swing_xz  = a_swing.linear[[0, 2]]

    Jdotv_h = np.zeros(2)
    Jdotv_h[0] = Jdotv_swing_xz[0] - Jdotv_stance_xz[0]
    Jdotv_h[1] = Jdotv_swing_xz[1] - Jdotv_stance_xz[1]
    return Jdotv_h

def compute_Jdotv_c(model, data, q, v, stance_fid):
    """J̇_c*q̇ for stance foot constraint (x,z linear acceleration with qdd=0). Returns (2,) array."""
    pin.forwardKinematics(model, data, q, v, np.zeros(model.nv))
    pin.updateFramePlacements(model, data)
    a6 = pin.getFrameClassicalAcceleration(model, data, stance_fid, pin.LOCAL_WORLD_ALIGNED)
    return a6.linear[[0, 2]]

def compute_M_and_h(model, data, q, v):
    """Mass matrix M and nonlinear effects h = C(q,v)v + g(q). Dynamics: M qdd + h = tau."""
    pin.crba(model, data, q)
    M = data.M.copy()
    # Pinocchio CRBA fills upper triangle only; symmetrize
    M = (M + M.T) - np.diag(np.diag(M))
    h = pin.nonLinearEffects(model, data, q, v)
    return M, h

def solve_constrained_forward_dynamics(M, h, Jc, Jcdotv, tau, b_baumgarte=None, reg=1e-8):
    """
    KKT: [M   -Jc.T] [qdd]   [tau - h  ]
         [Jc  -reg*I] [lam] = [RHS2    ]
    Constraint Jc q̈ + J̇c q̇ = 0. With optional Baumgarte:
      Jc q̈ + J̇c q̇ = -Kd(Jc q̇) - Kp φ(q)  =>  RHS2 = -J̇c q̇ - Kd(Jc q̇) - Kp φ(q).
    b_baumgarte = Kd*(Jc q̇) + Kp*φ(q), so RHS2 = -Jcdotv - b_baumgarte.
    """
    nv = M.shape[0]
    nc = Jc.shape[0]
    rhs1 = np.asarray(tau, dtype=float).ravel() - h
    rhs2 = -np.asarray(Jcdotv, dtype=float).ravel()  # -J̇c q̇
    if b_baumgarte is not None:
        rhs2 = rhs2 - np.asarray(b_baumgarte, dtype=float).ravel()  # RHS2 = -J̇c q̇ - Kd(Jc q̇) - Kp φ(q)
    KKT = np.block([[M, -Jc.T], [Jc, -reg * np.eye(nc)]])
    rhs = np.hstack([rhs1, rhs2])
    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    qdd = sol[:nv]
    lam = sol[nv:]
    return qdd, lam

def control_law(D, H, J, Jdotv, ydd_des, B, reg=1e-8, reg_M=1e-10):
    """
    Core equation (3): u = (J D⁻¹ B)⁻¹ (ÿ_d − J̇ q̇ + J D⁻¹ H).
    D = M, H = nle. Use solve on D+reg_M*I for stability; lstsq for pseudoinverse to avoid NaN.
    Returns u (nu,) and τ_full = B u (nv,).
    """
    nv = D.shape[0]
    D_safe = D + reg_M * np.eye(nv)
    Dinv_B = np.linalg.solve(D_safe, B)
    Dinv_H = np.linalg.solve(D_safe, np.asarray(H, dtype=float).ravel())
    A = J @ Dinv_B  # (ny x nu)
    rhs = np.asarray(ydd_des, dtype=float).ravel() - np.asarray(Jdotv, dtype=float).ravel() + (J @ Dinv_H)
    rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
    ny, nu = A.shape
    if ny <= nu:
        # (A A.T + reg I) x = rhs  =>  u = A.T x
        x, _, _, _ = np.linalg.lstsq(A @ A.T + reg * np.eye(ny), rhs, rcond=None)
        u = A.T @ np.asarray(x, dtype=float).ravel()
    else:
        u, _, _, _ = np.linalg.lstsq(A.T @ A + reg * np.eye(nu), A.T @ rhs, rcond=None)
        u = np.asarray(u, dtype=float).ravel()
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    tau_full = B @ u
    return u, tau_full

def stack_solve(Jh, Jc, Jcdotv, Jhdotv, hdd_des, damping=1e-6, b_contact_extra=None):
    """
    Solve [J; Jc] q̈ = [task_rhs; contact_rhs]. Contact: Jc q̈ + J̇c q̇ = -Kd(Jc q̇) - Kp φ(q)
    => contact_rhs = -Jcdotv - b_contact_extra with b_contact_extra = Kd*(Jc q̇) + Kp*φ(q).
    Uses damped least-squares (A.T A + damping*I) for numerical stability.
    """
    b_contact = -np.asarray(Jcdotv, dtype=float).ravel()
    if b_contact_extra is not None:
        b_contact = b_contact - np.asarray(b_contact_extra, dtype=float).ravel()
    A = np.vstack([Jh, Jc])
    b = np.hstack([hdd_des - Jhdotv, b_contact])

    nv = A.shape[1]
    qdd = np.linalg.solve(
        A.T @ A + damping * np.eye(nv),
        A.T @ b
    )
    return qdd

def torso_pitch_from_R(R):
    """Extract torso pitch angle (rotation about world y) from rotation matrix R; x-axis in world xz."""
    x_axis_W = R[:, 0]
    return np.arctan2(x_axis_W[2], x_axis_W[0])

def compute_h_hd(model, data, q, v, stance_fid, swing_fid):
    """
    Output h = swing foot position (x,z) relative to stance foot in world.
    Returns h (2,), hdot (2,), and Jacobian Jh (2, nv) such that hdot = Jh @ v.
    """
    pin.forwardKinematics(model, data, q, v)
    pin.updateFramePlacements(model, data)

    oMf_stance = data.oMf[stance_fid]
    oMf_swing  = data.oMf[swing_fid]
    p_stance_W = oMf_stance.translation
    p_swing_W  = oMf_swing.translation

    swing_rel_W = p_swing_W - p_stance_W
    x_rel = swing_rel_W[0]
    z_rel = swing_rel_W[2]

    h = np.array([x_rel, z_rel], dtype=float)

    J6_stance = pin.computeFrameJacobian(model, data, q, stance_fid, pin.LOCAL_WORLD_ALIGNED)
    J6_swing  = pin.computeFrameJacobian(model, data, q, swing_fid,  pin.LOCAL_WORLD_ALIGNED)
    Jlin_stance_xz = J6_stance[0:3, :][[0, 2], :]
    Jlin_swing_xz  = J6_swing[0:3, :][[0, 2], :]

    Jh = Jlin_swing_xz - Jlin_stance_xz
    hdot = Jh @ v
    return h, hdot, Jh

def simulate_one_step(model, data, q0, v0, a0, dt,
                      h_ref, hd_ref, hdd_des,
                      stance_fid, swing_fid, torso_fid,
                      v_com_des=0.25, kd_vcom=10.0, Kp=None, Kd=None,
                      p_stance_ref=None, kd_contact=50.0, kp_contact=500.0):
    """
    Output y = [swing xz rel stance, COM x rel stance]; J and J̇ q̇ are for these relative outputs.
    COM x acceleration follows basic H-LIP on level ground: xdd = (g/z)*x (LIP dynamics).
    (1) ÿ_d from ref + PD.  (2) q̈ from [J; Jc] q̈ = [ÿ_d − J̇ q̇; −J̇c q̇ − b] (task + foot constraint).
    (3) u from constrained dynamics: M q̈ + H = B u + Jc.T λ  =>  [B Jc.T] [u; λ] = M q̈ + h; integrate q̈.
    """
    if Kp is None:
        Kp = np.array([200.0, 400.0, 50.0])
    if Kd is None:
        Kd = np.array([30.0, 40.0, 10.0])

    pin.forwardKinematics(model, data, q0, v0)
    pin.updateFramePlacements(model, data)

    # Task and contact Jacobians / Jdot*v at current state
    h, hdot, Jh = compute_h_hd(model, data, q0, v0, stance_fid, swing_fid)
    Jc = compute_Jc(model, data, q0, v0, stance_fid)
    Jcdotv = compute_Jdotv_c(model, data, q0, v0, stance_fid)

    # Optional Baumgarte stabilization: Jc q̈ + J̇c q̇ = -Kd(Jc q̇) - Kp φ(q)
    b_contact_extra = None
    if p_stance_ref is not None:
        v_stance_xz = Jc @ v0                              # Jc q̇
        p_stance = stance_foot_world_xz(model, data, q0, stance_fid)
        phi_q = p_stance - p_stance_ref                    # φ(q)
        b_contact_extra = kd_contact * v_stance_xz + kp_contact * phi_q  # Kd(Jc q̇) + Kp φ(q)

    Jhdotv = compute_Jdotv_h(model, data, q0, v0, stance_fid, swing_fid, torso_fid)
    pin.centerOfMass(model, data, q0, v0)
    Jcom = pin.jacobianCenterOfMass(model, data, q0)
    # COM x relative to stance: J_com_x_rel = J_com_x - J_stance_x (same as swing–stance for h)
    J_stance_lin = pin.computeFrameJacobian(model, data, q0, stance_fid, pin.LOCAL_WORLD_ALIGNED)[:3, :]  # (3, nv)
    J_com_x_rel = Jcom[0:1, :] - J_stance_lin[0:1, :]
    pin.forwardKinematics(model, data, q0, v0, np.zeros(model.nv))
    pin.centerOfMass(model, data, q0, v0, np.zeros(model.nv))
    a_com_x = float(data.acom[0][0])
    a_stance = pin.getFrameClassicalAcceleration(model, data, stance_fid, pin.LOCAL_WORLD_ALIGNED)
    Jdotv_com_x_rel = a_com_x - float(a_stance.linear[0])
    x_com_rel = com_pos_rel_stance_x(model, data, q0, stance_fid)
    xdot_com_rel = com_vel_rel_stance_x(model, data, q0, v0, stance_fid)
    # Basic H-LIP (level ground): xdd = (g/z)*x
    z_com = com_height_relative_to_stance_z(model, data, q0, stance_fid)
    z_com = max(z_com, 1e-3)
    g = 9.81
    xdd_com_des = (g / z_com) * x_com_rel

    # Output y, ẏ and reference y_des, ẏ_des, ÿ_des
    y = np.hstack([h, x_com_rel])
    ydot = np.hstack([hdot, xdot_com_rel])
    y_des = np.hstack([np.asarray(h_ref).ravel()[:2], 0.0])
    ydot_des = np.hstack([np.asarray(hd_ref).ravel()[:2], v_com_des])
    ydd_des = np.hstack([np.asarray(hdd_des).ravel()[:2], xdd_com_des])

    J = np.vstack([Jh, J_com_x_rel])
    Jdotv = np.hstack([Jhdotv, Jdotv_com_x_rel])
    M, nle = compute_M_and_h(model, data, q0, v0)

    # Step 1: desired output acceleration ÿ_d (from ref + PD)
    ydd_d = ydd_des - Kp * (y - y_des) - Kd * (ydot - ydot_des)
    if np.any(np.isnan(ydd_d)):
        raise RuntimeError("NaN at step 1: ydd_d")
    # Step 2: q̈ from task + contact (constraint-aware): [J; Jc] q̈ = [ÿ_d − J̇ q̇; −J̇c q̇ − b]
    nv = M.shape[0]
    B = np.vstack([np.zeros((3, nv - 3)), np.eye(nv - 3)])
    qdd = stack_solve(J, Jc, Jcdotv, Jdotv, ydd_d, damping=1e-6, b_contact_extra=b_contact_extra)
    if np.any(np.isnan(qdd)):
        raise RuntimeError("NaN at step 2: qdd (task+contact solve)")
    # Step 3: u (and λ) from constrained dynamics: M q̈ + H = B u + Jc.T λ  =>  [B Jc.T] [u; λ] = M q̈ + h
    BL = np.hstack([B, Jc.T])
    rhs_dyn = M @ qdd + nle
    ulam, _, _, _ = np.linalg.lstsq(BL, rhs_dyn, rcond=None)
    u = np.asarray(ulam[: B.shape[1]], dtype=float).ravel()
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    # Integrate
    v_next = v0 + qdd * dt
    q_next = pin.integrate(model, q0, v_next * dt)
    if np.any(np.isnan(q_next)) or np.any(np.isnan(v_next)):
        raise RuntimeError("NaN after integrate: q_next or v_next")
    return q_next, v_next, h, hdot, qdd

def project_new_stance_velocity_zero(model, data, q, v, stance_fid, eps=1e-6):
    """
    Enforce NEW stance foot linear velocity (x,z) = 0 in LOCAL_WORLD_ALIGNED.

    v <- v + dv, where dv is minimum-norm correction.
    """
    Jc = compute_Jc(model, data, q, v, stance_fid)   # (2,nv)
    rhs = Jc @ v                                     # (2,)

    JJ = Jc @ Jc.T
    dv = - Jc.T @ np.linalg.solve(JJ + eps*np.eye(2), rhs)
    return v + dv

def coth(x):
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    # coth(x) ~ 1/x for x->0; avoid 0/0 and overflow
    out = np.where(ax < 1e-10, np.where(x >= 0, 1e10, -1e10), np.cosh(x) / np.sinh(x))
    return float(out) if np.isscalar(x) else out

def csch(x: float) -> float:
    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    out = np.where(ax < 1e-10, np.sign(x) * 1e10, 1.0 / np.sinh(x))
    return float(out) if np.isscalar(x) else out

def recompute_preimpact_deadbeat(Ts: float, Td: float, lam: float, vd: float):
    """
    H-LIP stepping stabilization per Xiong & Ames (TRO 2022).
    S2S state is PRE-IMPACT (end of SSP): x = [p^-, v^-]^T (COM rel. current stance).
    Stepping law: u = u* + K(x - x*)  (eq. 25).
    Deadbeat gain: K = [1, Td + (1/lam)*coth(Ts*lam)]  (eq. 27).

    Inputs
    ------
    Ts  : SSP duration (T_SSP)
    Td  : DSP duration (T_DSP)
    lam : sqrt(g/z0) pendulum constant
    vd  : desired net walking velocity

    Returns
    -------
    u_star       : nominal P1 step length = vd*(Ts+Td)
    x_star (2,)  : pre-impact fixed point [p^*, v^*] (eq. 20)
    K (1,2)      : deadbeat gain for u = u* + K(x - x*)
    """
    T = Ts + Td
    u_star = vd * T
    # Orbital slope sigma1 = lam * coth(Ts*lam/2), P1 fixed point (eq. 17, 20)
    sigma1 = lam * coth(lam * Ts / 2.0)
    p_star = u_star / (2.0 + Td * sigma1)
    v_star = sigma1 * p_star
    x_star = np.array([p_star, v_star], dtype=float)
    # Deadbeat gain (eq. 27): K = [1, Td + (1/lam)*coth(Ts*lam)]
    K = np.array([[1.0, Td + (1.0 / lam) * coth(lam * Ts)]], dtype=float)
    return u_star, x_star, K

# ==============================
# Plot Function
# ==============================
def plot_swing_foot_tracking_error(t_hist, h_hist, h_des_hist, show=True, save_path=None):
    """
    Plot error between desired h_des (reference swing foot x,z rel. stance) and
    actual h(q) (swing foot x,z rel. stance) for x and z.

    Parameters
    ----------
    t_hist : array-like
        Time vector (s), length N.
    h_hist : array-like, shape (N, 2)
        Actual h(q) = [x_swing_rel, z_swing_rel] at each time.
    h_des_hist : array-like, shape (N, 2)
        Desired h_des = [x_des, z_des] at each time.
    show : bool
        If True, call plt.show().
    save_path : str or None
        If set, save figure to this path.
    """
    t_hist = np.asarray(t_hist).ravel()
    h_hist = np.asarray(h_hist)
    h_des_hist = np.asarray(h_des_hist)
    if h_hist.ndim == 1:
        h_hist = h_hist.reshape(-1, 2)
    if h_des_hist.ndim == 1:
        h_des_hist = h_des_hist.reshape(-1, 2)

    err_x = h_hist[:, 0] - h_des_hist[:, 0]
    err_z = h_hist[:, 1] - h_des_hist[:, 1]

    fig, (ax_x, ax_z) = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax_x.plot(t_hist, err_x * 1000, "b-", linewidth=1.2, label=r"$h_x - h_{des,x}$")
    ax_x.set_ylabel("Error x (mm)")
    ax_x.legend(loc="best")
    ax_x.grid(True, alpha=0.3)

    ax_z.plot(t_hist, err_z * 1000, "r-", linewidth=1.2, label=r"$h_z - h_{des,z}$")
    ax_z.set_ylabel("Error z (mm)")
    ax_z.set_xlabel("Time (s)")
    ax_z.legend(loc="best")
    ax_z.grid(True, alpha=0.3)

    fig.suptitle("Swing foot tracking error: h(q) − h_des (x and z)")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_full_order_motion(model, q_traj, dt,
                           torso_name="torso",
                           right_leg=("right_thigh", "right_shin", "right_foot_point"),
                           left_leg =("left_thigh",  "left_shin",  "left_foot_point"),
                           stride=2, show=True, save_path=None):

    data = model.createData()


    torso_fid = model.getFrameId(torso_name)
    r_thigh_fid = model.getFrameId(right_leg[0])
    r_shin_fid  = model.getFrameId(right_leg[1])
    r_foot_fid  = model.getFrameId(right_leg[2])
    l_thigh_fid = model.getFrameId(left_leg[0])
    l_shin_fid  = model.getFrameId(left_leg[1])
    l_foot_fid  = model.getFrameId(left_leg[2])

    q_traj = list(q_traj)

    all_pts = []
    for k in range(0, len(q_traj), stride):
        qk = q_traj[k]
        pin.forwardKinematics(model, data, qk)
        pin.updateFramePlacements(model, data)
        pin.centerOfMass(model, data, qk)     # <-- add this
        pc = data.com[0][[0, 2]]              # <-- COM x,z

        p_t  = data.oMf[torso_fid].translation[[0, 2]]
        p_rt = data.oMf[r_thigh_fid].translation[[0, 2]]
        p_rs = data.oMf[r_shin_fid].translation[[0, 2]]
        p_rf = data.oMf[r_foot_fid].translation[[0, 2]]
        p_lt = data.oMf[l_thigh_fid].translation[[0, 2]]
        p_ls = data.oMf[l_shin_fid].translation[[0, 2]]
        p_lf = data.oMf[l_foot_fid].translation[[0, 2]]

        all_pts.append(np.vstack([p_t, p_rt, p_rs, p_rf, p_lt, p_ls, p_lf, pc]))

    all_pts = np.vstack(all_pts)
    xmin, xmax = float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0]))
    zmin, zmax = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))
    pad_x = 0.2 * max(1e-6, xmax - xmin)
    pad_z = 0.2 * max(1e-6, zmax - zmin)

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(zmin - pad_z, zmax + pad_z)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    title = ax.set_title("Full-order motion")

    (right_line,) = ax.plot([], [], "-", linewidth=2)
    (left_line,)  = ax.plot([], [], "-", linewidth=2)
    (com_pt,) = ax.plot([], [], "o", markersize=4)
    (torso_pt,) = ax.plot([], [], "o", markersize=4)
    (rfoot_pt,) = ax.plot([], [], "o", markersize=4)
    (lfoot_pt,) = ax.plot([], [], "o", markersize=4)

    frames = list(range(0, len(q_traj), stride))

    def init():
        right_line.set_data([], [])
        left_line.set_data([], [])
        torso_pt.set_data([], [])
        rfoot_pt.set_data([], [])
        lfoot_pt.set_data([], [])
        com_pt.set_data([], [])
        title.set_text("Full-order motion")
        return right_line, left_line, torso_pt, rfoot_pt, lfoot_pt, title

    def update(frame_idx):
        qk = q_traj[frame_idx]
        pin.forwardKinematics(model, data, qk)
        pin.centerOfMass(model, data, qk)
        pc = data.com[0]
        com_pt.set_data([float(pc[0])], [float(pc[2])])

        pin.updateFramePlacements(model, data)

        p_t  = data.oMf[torso_fid].translation[[0, 2]]
        p_rt = data.oMf[r_thigh_fid].translation[[0, 2]]
        p_rs = data.oMf[r_shin_fid].translation[[0, 2]]
        p_rf = data.oMf[r_foot_fid].translation[[0, 2]]
        p_lt = data.oMf[l_thigh_fid].translation[[0, 2]]
        p_ls = data.oMf[l_shin_fid].translation[[0, 2]]
        p_lf = data.oMf[l_foot_fid].translation[[0, 2]]

        right_chain = np.vstack([p_t, p_rt, p_rs, p_rf])
        left_chain  = np.vstack([p_t, p_lt, p_ls, p_lf])

        right_line.set_data(right_chain[:, 0], right_chain[:, 1])
        left_line.set_data(left_chain[:, 0], left_chain[:, 1])

        torso_pt.set_data([p_t[0]], [p_t[1]])
        rfoot_pt.set_data([p_rf[0]], [p_rf[1]])
        lfoot_pt.set_data([p_lf[0]], [p_lf[1]])

        tsec = frame_idx * dt
        title.set_text(f"Full-order motion | t = {tsec:.3f} s")
        return right_line, left_line, torso_pt, rfoot_pt, lfoot_pt, com_pt, title


    anim = FuncAnimation(fig, update, frames=frames,
                         init_func=init,
                         interval=1000*dt*stride, blit=True)

    if save_path is not None:
        anim.save(save_path, dpi=150)

    if show:
        plt.show()

    return anim


# ==============================
# Stance check
# ==============================
def stance_foot_world_xz(model, data, q, stance_fid):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p = data.oMf[stance_fid].translation
    return np.array([p[0], p[2]], dtype=float)


def check_stance_pinned(model, q_hist, v_hist, dt, stance_fid,
                        tol_pos=1e-3, tol_vel=1e-3, print_top=10):

    data = model.createData()

    P = np.zeros((len(q_hist), 2))
    for k, q in enumerate(q_hist):
        P[k] = stance_foot_world_xz(model, data, q, stance_fid)

    P0 = P[0].copy()
    dP = P - P0
    drift = np.linalg.norm(dP, axis=1)

    max_k = int(np.argmax(drift))
    print("STANCE POSITION DRIFT (world xz)")
    print("  p0 =", P0)
    print("  max drift =", drift[max_k], "m at k =", max_k,
          " t =", max_k*dt, "s")
    print("  p(k) =", P[max_k], "  dp =", dP[max_k])

    bad_pos = np.where(drift > tol_pos)[0]
    if len(bad_pos) > 0:
        print(f"  FAIL pos tol {tol_pos} m: {len(bad_pos)} samples exceed tol.")
        worst = np.argsort(-drift)[:print_top]
        for i in worst:
            print(f"    {i:4d}  {i*dt:8.4f}  {drift[i]:.6e}")
    else:
        print(f"  PASS pos tol {tol_pos} m")

    if v_hist is None:
        return

    Vfoot = np.zeros((len(v_hist), 2))
    for k, (q, v) in enumerate(zip(q_hist[:len(v_hist)], v_hist)):
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        J6 = pin.computeFrameJacobian(model, data, q,
                                      stance_fid,
                                      pin.LOCAL_WORLD_ALIGNED)
        Jlin_xz = J6[0:3, :][[0, 2], :]
        Vfoot[k] = Jlin_xz @ v

    speed = np.linalg.norm(Vfoot, axis=1)
    max_kv = int(np.argmax(speed))
    print("\nSTANCE VELOCITY (world xz)")
    print("  max |v_foot| =", speed[max_kv],
          "m/s at k =", max_kv, " t =", max_kv*dt, "s")
    print("  v_foot(k) =", Vfoot[max_kv])

    bad_vel = np.where(speed > tol_vel)[0]
    if len(bad_vel) > 0:
        print(f"  FAIL vel tol {tol_vel} m/s: {len(bad_vel)} samples exceed tol.")
        worst = np.argsort(-speed)[:print_top]
        for i in worst:
            print(f"    {i:4d}  {i*dt:8.4f}  {speed[i]:.6e}")
    else:
        print(f"  PASS vel tol {tol_vel} m/s")


# ==============================
# Main Function
# ==============================
def main():
    ensure_model_available()

    add_point_frame(model, "right_shin", "right_foot_point", [0.0, 0.0, 0.4])
    add_point_frame(model, "left_shin",  "left_foot_point",  [0.0, 0.0, 0.4])

    data = model.createData()
    qdd = np.zeros(model.nv)
    
    z_land = 0.0
    z_peak = 0.12
    Ts = 0.5
    Td = 0.1
    T = Ts + Td
    num_steps = 20
    vd = 0.3
    g = 9.81
    # Use standard gravity in model so full-order dynamics match H-LIP (g/z)*x
    model.gravity.linear[:] = np.array([0.0, 0.0, -g])
    ud = float(vd * (Ts + Td))  # nominal step length = 0.18 m for vd=0.3, T=0.6

    q = pin.neutral(model)
    v = np.zeros(model.nv)
    dt = 0.005
    t = np.linspace(0, Ts, int(Ts/dt) + 1)

    stance_fid = model.getFrameId("right_foot_point")
    swing_fid  = model.getFrameId("left_foot_point")
    torso_fid  = model.getFrameId("torso")

    # Joint angles set so z_com (CoM above stance foot) ≈ 0.75 m (tuned offline)
    set_continuous_joint_angle_in_q(model, q, "q1_left",  np.pi - 0.3)
    set_continuous_joint_angle_in_q(model, q, "q2_left",  1.23)
    set_continuous_joint_angle_in_q(model, q, "q1_right", np.pi - 0.3)
    set_continuous_joint_angle_in_q(model, q, "q2_right", 1.23)

    z_com = com_height_relative_to_stance_z(model, data, q, stance_fid)
    z_com = max(z_com, 1e-3)  # avoid sqrt(negative) or div by zero -> NaN
    lam = np.sqrt(g / z_com)

    # ---- STORE HISTORY FOR PLOTTING ----
    q_hist = []
    v_hist = []
    h_hist = []
    h_des_hist = []
    t_hist = []

    ud, x_des, K = recompute_preimpact_deadbeat(Ts, Td, lam, vd)
    uk = ud
    

    for i in range(num_steps):
        # Actual swing-relative state for tracking
        h0, hd0, _ = compute_h_hd(model, data, q, v, stance_fid, swing_fid)
        x_swing0, z_swing0 = h0
        h  = h0.copy()
        hd = hd0.copy()

        
        print(f"\nStep {i+1}/{num_steps} | HLIP step length uk = {uk:.4f} m")

        # --- rebuild Bezier for THIS step ---
        Px, Pz = build_step_bezier_points(x_swing0, z_swing0, xT= uk, zT=z_land, z_peak=z_peak)
        x, xd, xdd = bezier(t, 0, Ts, Px)
        z, zd, zdd = bezier(t, 0, Ts, Pz)
        # Pin stance foot: reference position (x, z) for Baumgarte stabilization
        p_stance_ref = stance_foot_world_xz(model, data, q, stance_fid)
        for k in range(len(t)):
            h_ref = np.array([x[k], z[k]])
            hd_ref = np.array([xd[k], zd[k]])
            hdd_ref = np.array([xdd[k], zdd[k]])

            q, v, h, hd, qdd = simulate_one_step(
                model, data, q, v, qdd, dt=dt,
                h_ref=h_ref, hd_ref=hd_ref, hdd_des=hdd_ref,
                stance_fid=stance_fid,
                swing_fid=swing_fid,
                torso_fid=torso_fid,
                v_com_des=vd,
                p_stance_ref=p_stance_ref
            )

            q_hist.append(q.copy())
            v_hist.append(v.copy())
            h_hist.append(h.copy())
            h_des_hist.append(h_ref.copy())
            t_hist.append(i * Ts + t[k])

        # PRE-IMPACT state (end of SSP): COM rel. current stance, before swap (paper eq. 25)
        p_pre = com_pos_rel_stance_x(model, data, q, stance_fid)
        v_pre = com_vel_rel_stance_x(model, data, q, v, stance_fid)  # signed
        x_pre = np.array([p_pre, v_pre])
        # Step length: u = u* + K(x - x*)  (paper eq. 25, 27)
        e_pre = x_pre - x_des
        uk = float(ud + (K @ e_pre)[0])


        print(f"  Pre-impact COM rel stance: p = {p_pre:.4f} m, v = {v_pre:.4f} m/s -> uk = {uk:.4f} m")

        # Swap legs for next step
        stance_fid, swing_fid = swing_fid, stance_fid
        v = project_new_stance_velocity_zero(model, data, q, v, stance_fid, eps=1e-4)
          


    # ---- PLOT / ANIMATE ----
    plot_full_order_motion(model, q_hist, dt, stride=2, show=True)
    plot_swing_foot_tracking_error(t_hist, h_hist, h_des_hist, show=True)



    
    
if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from None
