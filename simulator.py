"""
simulator.py

Orchestrates the attitude simulation loop.

Owns the satellite configuration, simulation state, and history.
Calls rigid_body.py each timestep. Controller and disturbances
will be plugged in here in later phases.

Usage
-----
  sim = Simulator(config)
  results = sim.run()
  sim.plot(results)
"""

import numpy as np
import matplotlib.pyplot as plt
from rigid_body import (
    build_inertia_tensor,
    quat_normalize,
    quat_to_rotation_matrix,
    rk4_step,
)


# ─────────────────────────────────────────────
#  Satellite Configuration
# ─────────────────────────────────────────────

class SatelliteConfig:
    """
    Holds all fixed physical properties of the satellite.
    Change these values to model different spacecraft.
    """

    def __init__(
        self,
        mass   = 10.0,          # kg
        lx     = 0.6,           # m — length along body X
        ly     = 0.2,           # m — length along body Y
        lz     = 0.2,           # m — length along body Z
    ):
        self.mass = mass
        self.lx   = lx
        self.ly   = ly
        self.lz   = lz

        # Derived quantities — computed once at init
        self.I     = build_inertia_tensor(mass, lx, ly, lz)
        self.I_inv = np.linalg.inv(self.I)

    def __repr__(self):
        return (
            f"SatelliteConfig(mass={self.mass}kg, "
            f"dims=[{self.lx},{self.ly},{self.lz}]m)\n"
            f"  Ixx={self.I[0,0]:.4f}, Iyy={self.I[1,1]:.4f}, Izz={self.I[2,2]:.4f} kg·m²"
        )


# ─────────────────────────────────────────────
#  Simulation Results Container
# ─────────────────────────────────────────────

class SimulationResults:
    """
    Holds all logged data from a completed simulation run.
    Passed to plotting functions.
    """

    def __init__(self, steps):
        self.t          = np.zeros(steps)           # time (s)
        self.q          = np.zeros((steps, 4))      # quaternion [w,x,y,z]
        self.omega      = np.zeros((steps, 3))      # angular velocity (rad/s)
        self.tau_ctrl   = np.zeros((steps, 3))      # control torque (N·m)
        self.tau_dist   = np.zeros((steps, 3))      # disturbance torque (N·m)
        self.tau_total  = np.zeros((steps, 3))      # total torque (N·m)
        self.error_deg  = np.zeros(steps)           # attitude error (degrees)

    def attitude_error_deg(self, q_desired):
        """
        Compute attitude error angle (degrees) at every timestep
        relative to a fixed desired quaternion.
        """
        from rigid_body import quat_multiply, quat_conjugate
        errors = np.zeros(len(self.t))
        for i, q in enumerate(self.q):
            q_err = quat_multiply(q_desired, np.array([q[0], -q[1], -q[2], -q[3]]))
            if q_err[0] < 0:
                q_err = -q_err
            angle = 2.0 * np.arccos(np.clip(abs(q_err[0]), 0.0, 1.0))
            errors[i] = np.degrees(angle)
        return errors


# ─────────────────────────────────────────────
#  Simulator
# ─────────────────────────────────────────────

class Simulator:
    """
    Main simulation driver.

    Parameters
    ----------
    config   : SatelliteConfig
    q0       : np.ndarray (4,)  — initial quaternion [w,x,y,z]
    omega0   : np.ndarray (3,)  — initial angular velocity (rad/s)
    dt       : float            — timestep (s), default 0.01
    t_end    : float            — simulation duration (s), default 120.0
    controller  : callable or None  — f(q, omega, t) → tau_ctrl (3,)
    disturbance : callable or None  — f(q, omega, t) → tau_dist (3,)
    """

    def __init__(
        self,
        config,
        q0          = None,
        omega0      = None,
        dt          = 0.01,
        t_end       = 120.0,
        controller  = None,
        disturbance = None,
    ):
        self.config      = config
        self.dt          = dt
        self.t_end       = t_end
        self.controller  = controller
        self.disturbance = disturbance

        # Default initial state: identity quaternion, no spin
        self.q0     = q0     if q0     is not None else np.array([1.0, 0.0, 0.0, 0.0])
        self.omega0 = omega0 if omega0 is not None else np.zeros(3)

    def run(self):
        """
        Execute the simulation loop.

        Returns
        -------
        results : SimulationResults
        """
        steps   = int(self.t_end / self.dt)
        results = SimulationResults(steps)
        cfg     = self.config

        q     = quat_normalize(self.q0.copy())
        omega = self.omega0.copy()

        for i in range(steps):
            t = i * self.dt

            # ── Compute torques ──────────────────────────────
            tau_ctrl = (
                self.controller(q, omega, t)
                if self.controller is not None
                else np.zeros(3)
            )

            tau_dist = (
                self.disturbance(q, omega, t)
                if self.disturbance is not None
                else np.zeros(3)
            )

            tau_total = tau_ctrl + tau_dist

            # ── Log current state (before stepping) ──────────
            results.t[i]         = t
            results.q[i]         = q
            results.omega[i]     = omega
            results.tau_ctrl[i]  = tau_ctrl
            results.tau_dist[i]  = tau_dist
            results.tau_total[i] = tau_total

            # ── Integrate one timestep ────────────────────────
            q, omega = rk4_step(q, omega, cfg.I, cfg.I_inv, tau_total, self.dt)

        print(f"Simulation complete — {steps} steps, {self.t_end:.1f}s simulated.")
        return results

    def plot(self, results, q_desired=None, title_suffix=""):
        """
        Generate a standard 4-panel diagnostic plot.

        Parameters
        ----------
        results    : SimulationResults
        q_desired  : np.ndarray (4,) or None — desired attitude for error plot
        title_suffix : str — appended to figure title
        """
        if q_desired is None:
            q_desired = np.array([1.0, 0.0, 0.0, 0.0])

        error_deg = results.attitude_error_deg(q_desired)
        t = results.t

        fig, axes = plt.subplots(4, 1, figsize=(11, 13))
        fig.suptitle(f"Attitude Simulation Results {title_suffix}", fontsize=13, y=0.98)

        # ── Panel 1: Angular velocity ─────────────────────────
        axes[0].plot(t, np.degrees(results.omega[:, 0]), color='#e84040', label='ωx')
        axes[0].plot(t, np.degrees(results.omega[:, 1]), color='#40c040', label='ωy')
        axes[0].plot(t, np.degrees(results.omega[:, 2]), color='#4080e8', label='ωz')
        axes[0].set_ylabel('Angular velocity (°/s)')
        axes[0].set_title('Angular velocity — body frame')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(0, color='gray', linewidth=0.5)

        # ── Panel 2: Quaternion components ────────────────────
        axes[1].plot(t, results.q[:, 0], color='#888888', label='w')
        axes[1].plot(t, results.q[:, 1], color='#e84040', label='x')
        axes[1].plot(t, results.q[:, 2], color='#40c040', label='y')
        axes[1].plot(t, results.q[:, 3], color='#4080e8', label='z')
        axes[1].set_ylabel('Quaternion components')
        axes[1].set_title('Attitude quaternion')
        axes[1].set_ylim(-1.1, 1.1)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(0, color='gray', linewidth=0.5)

        # ── Panel 3: Attitude error ───────────────────────────
        axes[2].plot(t, error_deg, color='#f0a020', linewidth=1.5)
        axes[2].set_ylabel('Attitude error (°)')
        axes[2].set_title('Pointing error from desired attitude')
        axes[2].set_ylim(bottom=0)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(0, color='gray', linewidth=0.5)

        # ── Panel 4: Applied torques ──────────────────────────
        tau_mag = np.linalg.norm(results.tau_total, axis=1) * 1000  # N·m → mN·m
        axes[3].plot(t, results.tau_total[:, 0] * 1000, color='#e84040', alpha=0.6, label='τx')
        axes[3].plot(t, results.tau_total[:, 1] * 1000, color='#40c040', alpha=0.6, label='τy')
        axes[3].plot(t, results.tau_total[:, 2] * 1000, color='#4080e8', alpha=0.6, label='τz')
        axes[3].plot(t, tau_mag,                         color='#f0a020', linewidth=1.5, label='|τ|')
        axes[3].set_ylabel('Torque (mN·m)')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_title('Applied torques')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        axes[3].axhline(0, color='gray', linewidth=0.5)

        plt.tight_layout()
        filename = f"sim_results{title_suffix.replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150)
        plt.show()
        print(f"Plot saved: {filename}")


# ─────────────────────────────────────────────
#  Quick Tests
# ─────────────────────────────────────────────

if __name__ == "__main__":

    cfg = SatelliteConfig(mass=10.0, lx=0.6, ly=0.2, lz=0.2)
    print(cfg)
    print()

    # ── Test 1: Free tumble (no torques) ─────────────────────
    print("=" * 50)
    print("Test 1: Free tumble — no controller, no disturbances")
    print("=" * 50)

    sim = Simulator(
        config  = cfg,
        omega0  = np.array([0.05, 0.30, 0.05]),   # mostly Y-axis spin
        t_end   = 60.0,
    )
    results = sim.run()
    sim.plot(results, title_suffix="— free tumble")

    # ── Test 2: Zero torque, zero spin — should stay still ───
    print()
    print("=" * 50)
    print("Test 2: Static satellite — should not move")
    print("=" * 50)

    sim2 = Simulator(
        config = cfg,
        q0     = np.array([1.0, 0.0, 0.0, 0.0]),
        omega0 = np.zeros(3),
        t_end  = 30.0,
    )
    results2 = sim2.run()

    max_drift = np.max(np.abs(results2.omega))
    final_q   = results2.q[-1]
    print(f"Max angular velocity drift: {max_drift:.2e} rad/s  (should be ≈ 0.0)")
    print(f"Final quaternion: {np.round(final_q, 8)}  (should be [1,0,0,0])")

    # ── Test 3: Constant torque — should spin up steadily ────
    print()
    print("=" * 50)
    print("Test 3: Constant torque about Z — should spin up linearly")
    print("=" * 50)

    tau_z       = np.array([0.0, 0.0, 0.01])   # 10 mN·m about Z
    expected_alpha = tau_z[2] / cfg.I[2, 2]     # α = τ / Izz

    sim3 = Simulator(
        config      = cfg,
        omega0      = np.zeros(3),
        t_end       = 30.0,
        controller  = lambda q, omega, t: tau_z,
    )
    results3 = sim3.run()

    final_omega_z  = results3.omega[-1, 2]
    expected_omega_z = expected_alpha * 30.0    # ω = α · t
    print(f"Expected final ωz: {np.degrees(expected_omega_z):.2f} °/s")
    print(f"Simulated final ωz: {np.degrees(final_omega_z):.2f} °/s")
    print(f"Error: {abs(final_omega_z - expected_omega_z):.2e} rad/s  (should be tiny)")