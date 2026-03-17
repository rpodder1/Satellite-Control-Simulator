"""
rigid_body.py

Implements the core rotational dynamics of a rigid spacecraft.

State vector: [q0, q1, q2, q3, wx, wy, wz]
  q = unit quaternion (scalar-first convention: [w, x, y, z])
  w = angular velocity in body frame (rad/s)
"""

import numpy as np


# ─────────────────────────────────────────────
#  Inertia Tensor
# ─────────────────────────────────────────────

def build_inertia_tensor(mass, lx, ly, lz):
    """
    Compute the inertia tensor for a uniform rectangular box.

    Parameters
    ----------
    mass : float        — total mass in kg
    lx, ly, lz : float  — side lengths in meters

    Returns
    -------
    I : np.ndarray, shape (3,3)  — inertia tensor in body frame (kg·m²)
    """
    Ixx = (mass / 12.0) * (ly**2 + lz**2)
    Iyy = (mass / 12.0) * (lx**2 + lz**2)
    Izz = (mass / 12.0) * (lx**2 + ly**2)

    return np.diag([Ixx, Iyy, Izz])


# ─────────────────────────────────────────────
#  Quaternion Utilities
# ─────────────────────────────────────────────

def quat_normalize(q):
    """
    Normalize a quaternion to unit length.
    Call after every integration step to prevent floating point drift.
    """
    return q / np.linalg.norm(q)


def quat_conjugate(q):
    """
    Conjugate of quaternion q = [w, x, y, z].
    For unit quaternions, conjugate == inverse.
    Represents the opposite rotation.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1, q2):
    """
    Hamilton product q1 ⊗ q2.
    Composes two rotations: first apply q2, then q1.

    Parameters
    ----------
    q1, q2 : np.ndarray (4,)  — unit quaternions [w, x, y, z]

    Returns
    -------
    q : np.ndarray (4,)  — composed rotation (not normalized — call quat_normalize if needed)
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_to_rotation_matrix(q):
    """
    Convert unit quaternion to 3x3 rotation matrix.
    Transforms vectors from body frame to inertial frame.

    Parameters
    ----------
    q : np.ndarray (4,)  — unit quaternion [w, x, y, z]

    Returns
    -------
    R : np.ndarray (3,3)  — rotation matrix
    """
    w, x, y, z = q

    return np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),       2*(x*z + w*y)  ],
        [    2*(x*y + w*z),     1 - 2*(x**2 + z**2),     2*(y*z - w*x) ],
        [    2*(x*z - w*y),         2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ])


# ─────────────────────────────────────────────
#  Equations of Motion
# ─────────────────────────────────────────────

def omega_dot(omega, I, I_inv, tau):
    """
    Euler's rotation equation.
    Computes angular acceleration given current state and applied torque.

    τ_total = I·ω̇ + ω × (I·ω)
    → ω̇ = I⁻¹ · (τ - ω × (I·ω))

    Parameters
    ----------
    omega : np.ndarray (3,)   — angular velocity in body frame (rad/s)
    I     : np.ndarray (3,3)  — inertia tensor (kg·m²)
    I_inv : np.ndarray (3,3)  — precomputed inverse of I (compute once, reuse in RK4)
    tau   : np.ndarray (3,)   — total applied torque in body frame (N·m)

    Returns
    -------
    domega_dt : np.ndarray (3,)  — angular acceleration (rad/s²)
    """
    gyroscopic = np.cross(omega, I @ omega)
    return I_inv @ (tau - gyroscopic)


def q_dot(q, omega):
    """
    Quaternion kinematics.
    Computes rate of change of quaternion given angular velocity.

    q̇ = ½ · Ω(ω) · q

    Parameters
    ----------
    q     : np.ndarray (4,)  — current attitude quaternion [w, x, y, z]
    omega : np.ndarray (3,)  — angular velocity in body frame (rad/s)

    Returns
    -------
    dq_dt : np.ndarray (4,)  — quaternion derivative
    """
    p, q_r, r = omega   # q_r to avoid shadowing the quaternion variable q

    Omega = 0.5 * np.array([
        [ 0,   -p,  -q_r,  -r  ],
        [ p,    0,    r,   -q_r ],
        [q_r,  -r,    0,    p  ],
        [ r,   q_r,  -p,    0  ]
    ])

    return Omega @ q


# ─────────────────────────────────────────────
#  RK4 Integrator
# ─────────────────────────────────────────────

def rk4_step(q, omega, I, I_inv, tau, dt):
    """
    Advance the rotational state by one timestep using 4th-order Runge-Kutta.

    Integrates both omega_dot and q_dot simultaneously.
    Torque tau is assumed constant over the timestep (zero-order hold).

    Parameters
    ----------
    q     : np.ndarray (4,)   — current quaternion [w, x, y, z]
    omega : np.ndarray (3,)   — current angular velocity (rad/s)
    I     : np.ndarray (3,3)  — inertia tensor
    I_inv : np.ndarray (3,3)  — inverse inertia tensor
    tau   : np.ndarray (3,)   — total torque in body frame (N·m)
    dt    : float             — timestep in seconds

    Returns
    -------
    q_new     : np.ndarray (4,)  — updated quaternion (normalized)
    omega_new : np.ndarray (3,)  — updated angular velocity (rad/s)
    """

    # k1 — derivatives at current state
    k1_omega = omega_dot(omega,                          I, I_inv, tau)
    k1_q     = q_dot(q,                                  omega)

    # k2 — derivatives at midpoint using k1 estimates
    omega_2  = omega + 0.5 * dt * k1_omega
    q_2      = quat_normalize(q + 0.5 * dt * k1_q)
    k2_omega = omega_dot(omega_2,                        I, I_inv, tau)
    k2_q     = q_dot(q_2,                                omega_2)

    # k3 — derivatives at midpoint using k2 estimates
    omega_3  = omega + 0.5 * dt * k2_omega
    q_3      = quat_normalize(q + 0.5 * dt * k2_q)
    k3_omega = omega_dot(omega_3,                        I, I_inv, tau)
    k3_q     = q_dot(q_3,                                omega_3)

    # k4 — derivatives at end using k3 estimates
    omega_4  = omega + dt * k3_omega
    q_4      = quat_normalize(q + dt * k3_q)
    k4_omega = omega_dot(omega_4,                        I, I_inv, tau)
    k4_q     = q_dot(q_4,                                omega_4)

    # Weighted average of all four estimates
    omega_new = omega + (dt / 6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
    q_new     = q     + (dt / 6.0) * (k1_q     + 2*k2_q     + 2*k3_q     + k4_q    )

    # Normalize quaternion — prevents floating point drift off the unit sphere
    q_new = quat_normalize(q_new)

    return q_new, omega_new


# ─────────────────────────────────────────────
#  Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Build satellite ---
    mass      = 10.0              # kg
    lx, ly, lz = 0.6, 0.2, 0.2   # meters — elongated along X
    I         = build_inertia_tensor(mass, lx, ly, lz)
    I_inv     = np.linalg.inv(I)

    print("Inertia tensor (kg·m²):")
    print(np.round(I, 4))
    print()

    # --- Initial state ---
    # Spin mostly about the intermediate axis (Y) — should produce chaotic tumbling
    q     = np.array([1.0, 0.0, 0.0, 0.0])   # identity quaternion — no initial rotation
    omega = np.array([0.05, 0.30, 0.05])       # rad/s
    tau   = np.zeros(3)                         # no torques — pure free tumble

    # --- Simulation parameters ---
    dt    = 0.01     # 10 ms timestep
    t_end = 60.0     # seconds
    steps = int(t_end / dt)

    # --- History arrays ---
    t_hist     = np.zeros(steps)
    omega_hist = np.zeros((steps, 3))
    q_hist     = np.zeros((steps, 4))

    # --- Run simulation ---
    for i in range(steps):
        t_hist[i]     = i * dt
        omega_hist[i] = omega
        q_hist[i]     = q

        q, omega = rk4_step(q, omega, I, I_inv, tau, dt)

    # --- Plot results ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    axes[0].plot(t_hist, np.degrees(omega_hist[:, 0]), color='#e84040', label='ωx')
    axes[0].plot(t_hist, np.degrees(omega_hist[:, 1]), color='#40c040', label='ωy')
    axes[0].plot(t_hist, np.degrees(omega_hist[:, 2]), color='#4080e8', label='ωz')
    axes[0].set_ylabel('Angular velocity (°/s)')
    axes[0].set_title('Free tumble — no torques (intermediate axis theorem)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_hist, q_hist[:, 0], color='#888888', label='w')
    axes[1].plot(t_hist, q_hist[:, 1], color='#e84040', label='x')
    axes[1].plot(t_hist, q_hist[:, 2], color='#40c040', label='y')
    axes[1].plot(t_hist, q_hist[:, 3], color='#4080e8', label='z')
    axes[1].set_ylabel('Quaternion components')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Attitude evolution during free tumble')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_free_tumble.png', dpi=150)
    plt.show()
    print("Plot saved: test_free_tumble.png")

    # --- Sanity checks ---
    norms = np.linalg.norm(q_hist, axis=1)
    print(f"Quaternion norm — min: {norms.min():.8f}, max: {norms.max():.8f}  (should be ≈ 1.0)")

    # Stable spin test — spin purely about Z (major axis), should stay clean
    print("\nRunning stable spin test (pure Z-axis spin)...")
    q2    = np.array([1.0, 0.0, 0.0, 0.0])
    omega2 = np.array([0.0, 0.0, 1.0])
    for _ in range(steps):
        q2, omega2 = rk4_step(q2, omega2, I, I_inv, np.zeros(3), dt)
    print(f"After 60s of Z-spin — ωx: {omega2[0]:.6f}, ωy: {omega2[1]:.6f}  (should be ≈ 0.0)")
    print(f"                    — ωz: {omega2[2]:.6f}  (should be ≈ 1.0)")