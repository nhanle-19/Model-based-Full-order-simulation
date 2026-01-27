"""
Author: Nhan Le
Date: 2026-01-19
"""

# ==============================
# Imports
# ==============================
import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pinocchio as pin
import os

# ==============================
# Model definition
# ==============================
urdf = "five_link_walker.urdf"
model = pin.buildModelFromUrdf(urdf)
data = model.createData()           #create Pinocchio workspace

# ==============================
# Functions
# ==============================
def joint_idx_q(model, joint_name):  #get index of joint in q vector
    jid = model.getJointId(joint_name)
    return model.joints[jid].idx_q

def set_continuous_joint_angle_in_q(model, q, joint_name, theta): #update continuous joint back in q vector
    i = joint_idx_q(model, joint_name)
    q[i]   = np.cos(theta)
    q[i+1] = np.sin(theta)

def joint_idx_v(model, joint_name): #get index of joint in v vector
    jid = model.getJointId(joint_name)
    return model.joints[jid].idx_v

def get_continuous_joint_angle_from_q(model, q, joint_name): #extract continuous joint angle from q vector
    i = joint_idx_q(model, joint_name)
    return np.arctan2(q[i+1], q[i])

def add_point_frame(model, parent_frame_name, new_frame_name, xyz_local):
    """
    Add a frame rigidly attached to an existing frame.

    Parameters
    ----------
    model : pin.Model
        Pinocchio model
    parent_frame_name : str
        Name of existing frame (e.g. "right_shin")
    new_frame_name : str
        Name of the new frame (e.g. "right_foot_point")
    xyz_local : array-like, shape (3,)
        Position of new frame expressed in parent frame coordinates
    """

    # 1. Get parent frame
    parent_fid = model.getFrameId(parent_frame_name)
    parent_frame = model.frames[parent_fid]

    # 2. Joint that moves this frame
    parent_joint = parent_frame.parentJoint

    # 3. Transform: parent frame -> new frame
    X_parent_new = pin.SE3(np.eye(3), np.array(xyz_local))

    # 4. Express new frame placement relative to the joint
    X_joint_new = parent_frame.placement * X_parent_new

    # 5. Create and add the frame
    new_frame = pin.Frame(
        new_frame_name,
        parent_joint,
        parent_fid,
        X_joint_new,
        pin.FrameType.OP_FRAME
    )

    model.addFrame(new_frame)

def mass_matrix(q):
    pin.crba(model, data, q)
    # Pinocchio fills only upper triangular; symmetrize:
    M = data.M.copy()
    M = (M + M.T) - np.diag(np.diag(M))
    return M

def nonlinear_effects(q, v):
    h = pin.nonLinearEffects(model, data, q, v)  # nv vector: C(q,v)v + g(q)
    return h.copy()

def forward_dynamics_zero_torque(M, h):
    qdd = np.linalg.solve(M, -h) #for zero input
    return qdd

def qdot_from_sincos(q, v):
    """
    q: shape (8,) = [c1,s1,c2,s2,c3,s3,c4,s4]
    v: shape (4,) = [th1dot, th2dot, th3dot, th4dot]
    returns qdot: shape (8,)
    """
    qdot = np.zeros_like(q)

    for k in range(4):
        c = q[2*k]
        s = q[2*k + 1]
        thdot = v[k]

        qdot[2*k]     = -s * thdot
        qdot[2*k + 1] =  c * thdot

    return qdot


def vector_field(t, x):
    x = np.asarray(x).reshape(-1)
    q = x[:model.nq]        # (8,)
    v = x[model.nq:]        # (4,)

    # Dynamics (Pinocchio)
    D = mass_matrix(q)
    h = nonlinear_effects(q, v)
    qdd = forward_dynamics_zero_torque(D, h)   # (4,) because nv=4

    # Kinematics for sin/cos q
    qdot = qdot_from_sincos(q, v)              # (8,)

    # Return same length as x: (8 + 4) = 12
    return np.hstack((qdot, qdd))


# --- Plotting function (sagittal stick figure) ---
def plot_full_order_sagittal_stick_v2(
    q,
    urdf_path,
    torso_frame_name="torso",
    torso_keyword="torso",
    left_hip_joint="q1_left",
    left_knee_joint="q2_left",
    right_hip_joint="q1_right",
    right_knee_joint="q2_right",
    distal_keywords=("foot", "ankle", "toe", "shin"),
    separate_legs_for_visibility=True,
    sep=0.03,
    show_labels=True,
):
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    q = np.asarray(q, float).reshape(-1)
    if q.size != model.nq:
        raise ValueError(f"q size {q.size} != model.nq {model.nq}")

    mesh_dir = os.path.dirname(urdf_path)
    geom_model = pin.buildGeomFromUrdf(model, urdf_path, pin.GeometryType.VISUAL, mesh_dir)
    geom_data = pin.GeometryData(geom_model)

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pin.updateGeometryPlacements(model, data, geom_model, geom_data)

    # --- torso axes (sagittal) ---
    fid = model.getFrameId(torso_frame_name)
    if fid == len(model.frames):
        R = np.eye(3)
        p_torso = np.zeros(3)
    else:
        oMf = data.oMf[fid]
        R = oMf.rotation.copy()
        p_torso = oMf.translation.copy()

    ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
    eu = ez  # up

    # --- torso top anchor (highest torso visual along eu) ---
    torso_ids = [i for i, go in enumerate(geom_model.geometryObjects)
                 if torso_keyword.lower() in go.name.lower()]
    if torso_ids:
        best_i = max(torso_ids, key=lambda i: float(eu.dot(geom_data.oMg[i].translation)))
        torso_top = geom_data.oMg[best_i].translation.copy()
    else:
        torso_top = p_torso.copy()

    origin = torso_top  # (0,0) will be torso_top in sagittal plot

    def jpos(jname):
        jid = model.getJointId(jname)
        if jid == 0:
            raise ValueError(f"Joint '{jname}' not found in URDF.")
        return data.oMi[jid].translation.copy()

    Lhip = jpos(left_hip_joint)
    Rhip = jpos(right_hip_joint)
    Lknee = jpos(left_knee_joint)
    Rknee = jpos(right_knee_joint)

    # NEW: single pelvis point that "joins" the hips
    pelvis = 0.5 * (Lhip + Rhip)

    # --- find distal point from visuals (simple + works with your shin visuals) ---
    def find_distal(side, ref):
        best_p, best_name, best_d = None, None, -1.0
        for i, go in enumerate(geom_model.geometryObjects):
            nm = go.name.lower()
            if side not in nm:
                continue
            if not any(k in nm for k in distal_keywords):
                continue
            p = geom_data.oMg[i].translation
            d = np.linalg.norm(p - ref)
            if d > best_d:
                best_d, best_p, best_name = d, p.copy(), go.name
        return best_p, best_name

    Ldist, Ld_name = find_distal("left", Lknee)
    Rdist, Rd_name = find_distal("right", Rknee)
    if Ldist is None or Rdist is None:
        raise ValueError("Couldn't find distal points. Try adding 'foot' or 'ankle' visuals/frames in URDF.")

    # --- choose forward axis to keep thighs visible ---
    def proj_len(ef):
        def uv(p):
            d = p - origin
            return np.array([ef.dot(d), eu.dot(d)])
        # use pelvis->knees for thigh visibility now
        return np.linalg.norm(uv(Lknee) - uv(pelvis)) + np.linalg.norm(uv(Rknee) - uv(pelvis))

    ef = ex if proj_len(ex) >= proj_len(ey) else ey

    def uv(p):
        d = p - origin
        return np.array([ef.dot(d), eu.dot(d)])

    # project
    pelvis_uv = uv(pelvis)
    Lk_uv = uv(Lknee);  Rk_uv = uv(Rknee)
    Ld_uv = uv(Ldist);  Rd_uv = uv(Rdist)
    top_uv = np.array([0.0, 0.0])  # torso_top at origin

    # optional: separate legs (ONLY knees/distals) so pelvis stays shared
    if separate_legs_for_visibility:
        Lk_uv[0] -= sep; Ld_uv[0] -= sep
        Rk_uv[0] += sep; Rd_uv[0] += sep

    # --- plot ---
    fig, ax = plt.subplots()

    # NEW: torso line pelvis -> torso_top
    ax.plot([pelvis_uv[0], top_uv[0]], [pelvis_uv[1], top_uv[1]], "-", linewidth=4)

    # left leg: pelvis->knee (thigh solid), knee->distal (shin dashed)
    ax.plot([pelvis_uv[0], Lk_uv[0]], [pelvis_uv[1], Lk_uv[1]], "-", linewidth=3)
    ax.plot([Lk_uv[0], Ld_uv[0]],     [Lk_uv[1], Ld_uv[1]],     "--", linewidth=3)

    # right leg
    ax.plot([pelvis_uv[0], Rk_uv[0]], [pelvis_uv[1], Rk_uv[1]], "-", linewidth=3)
    ax.plot([Rk_uv[0], Rd_uv[0]],     [Rk_uv[1], Rd_uv[1]],     "--", linewidth=3)

    # points
    ax.scatter([top_uv[0]], [top_uv[1]], s=140)
    ax.scatter([pelvis_uv[0]], [pelvis_uv[1]], s=110)
    ax.scatter([Lk_uv[0], Rk_uv[0]], [Lk_uv[1], Rk_uv[1]], s=90)
    ax.scatter([Ld_uv[0], Rd_uv[0]], [Ld_uv[1], Rd_uv[1]], s=90)

    if show_labels:
        ax.text(0.0, 0.0, " torso_top", fontsize=9)
        ax.text(pelvis_uv[0], pelvis_uv[1], " pelvis", fontsize=9)
        ax.text(Lk_uv[0], Lk_uv[1], " L_knee", fontsize=9)
        ax.text(Rk_uv[0], Rk_uv[1], " R_knee", fontsize=9)
        ax.text(Ld_uv[0], Ld_uv[1], f" L_distal({Ld_name})", fontsize=8)
        ax.text(Rd_uv[0], Rd_uv[1], f" R_distal({Rd_name})", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_xlabel("forward (torso sagittal)")
    ax.set_ylabel("up (torso)")
    ax.set_title("Sagittal stick (shared pelvis + torso line)")
    plt.show()

    return {"forward_axis": "torso_x" if np.allclose(ef, ex) else "torso_y",
            "L_distal": Ld_name, "R_distal": Rd_name}



# ==============================
# Main Function
# ==============================
def main():
    # ---- Constants / Parameters ----
    q = pin.neutral(model)

    add_point_frame(model, "right_shin", "right_foot_point", [0.0, 0.0, 0.4])
    add_point_frame(model, "left_shin",  "left_foot_point",  [0.0, 0.0, 0.4])

    set_continuous_joint_angle_in_q(model, q, "q1_left", 2*np.pi/3)
    set_continuous_joint_angle_in_q(model, q, "q2_left", 2*np.pi/5)
    set_continuous_joint_angle_in_q(model, q, "q1_right", 2*np.pi/3)
    set_continuous_joint_angle_in_q(model, q, "q2_right", 0.0)

    D = mass_matrix(q)
    h = nonlinear_effects(q, np.zeros(model.nv))
    qdd = forward_dynamics_zero_torque(D, h)
    plot_full_order_sagittal_stick_v2(q, urdf,sep=0.03)

    x = np.hstack((q, np.zeros(model.nv)))
    xd = np.hstack((np.zeros(model.nv), qdd))
    print("Initial q:", q)
    sol = solve_ivp(fun=vector_field,t_span=(0.0, 5),y0=x,method="RK45",max_step=1e-3,rtol=1e-6,atol=1e-9)
    
    x = sol.y[:, -1]
    q = x[:model.nq]
    print("Final q:",q)
   
    
    info = plot_full_order_sagittal_stick_v2(q, urdf,sep=0.03)

    
  
   

    # ---- Run solver ----
   

    # ---- Output ----
    



# ==============================
# Script Entry Point
# ==============================
if __name__ == "__main__":
    main()
