"""
pid_controller.py

PID attitude controller for a rigid spacecraft.

Eliminates steady-state pointing error against persistent disturbances
(SRP, aerodynamic drag) via integral action.

Control law:
    tau = Kp * q_err_vec + Ki * integral(q_err_vec) - Kd * omega

Gain design:
    Kp = wn^2 * I
    Kd = 2 * zeta * wn * I
    Ki = wn * ki_scale * I

    wn       = natural frequency (rad/s)  — controls response speed
    zeta     = damping ratio              — 0.7 = no overshoot, fast
    ki_scale = integral gain scale        — 0.125 is a safe starting point

Anti-windup:
    During large slews the integral accumulates a large value.
    When the satellite nears the target this causes overshoot.
    The integral torque magnitude is clamped to tau_max * 0.5
    so it can never dominate the proportional term mid-slew.
"""

import numpy as np
from rigid_body import quat_multiply, quat_conjugate


# ─────────────────────────────────────────────
#  Error quaternion utility
# ─────────────────────────────────────────────

def compute_error_quaternion(q, q_desired):
    """
    Compute error quaternion and enforce shortest-path rotation.

    Returns
    -------
    q_err   : np.ndarray (4,)  — error quaternion
    err_vec : np.ndarray (3,)  — vector part [x,y,z], used as error signal
    err_deg : float            — scalar error angle in degrees
    """
    q_err = quat_multiply(q_desired, quat_conjugate(q))

    # Shortest path: q and -q are the same rotation,
    # but -q takes the long way around (360-angle instead of angle)
    if q_err[0] < 0:
        q_err = -q_err

    err_vec = q_err[1:]
    err_deg = np.degrees(2.0 * np.arccos(np.clip(abs(q_err[0]), 0.0, 1.0)))

    return q_err, err_vec, err_deg


# ─────────────────────────────────────────────
#  PID Controller
# ─────────────────────────────────────────────

class PIDController:
    """
    Proportional-Integral-Derivative attitude controller.

    Parameters
    ----------
    I         : np.ndarray (3,3)  — inertia tensor (kg*m^2)
    wn        : float             — natural frequency (rad/s)
    zeta      : float             — damping ratio (0.7 recommended)
    ki_scale  : float             — Ki = wn * ki_scale * I
                                    increase for faster disturbance rejection
                                    decrease if you see oscillation
    tau_max   : float             — anti-windup clamp magnitude (N*m)
    q_desired : np.ndarray (4,)   — desired attitude quaternion
    dt        : float             — timestep (must match Simulator dt)
    """

    def __init__(
        self,
        I,
        wn        = 0.3,
        zeta      = 0.7,
        ki_scale  = 0.02,
        tau_max   = 0.005,
        q_desired = None,
        dt        = 0.05,
    ):
        self.I       = I
        self.wn      = wn
        self.zeta    = zeta
        self.tau_max = tau_max
        self.dt      = dt

        self.Kp = (wn**2)         * I
        self.Kd = (2 * zeta * wn) * I
        self.Ki = (wn * ki_scale) * I

        self.q_desired = (
            q_desired if q_desired is not None
            else np.array([1.0, 0.0, 0.0, 0.0])
        )

        # Integral accumulator
        self.integral = np.zeros(3)

        print(f"PIDController initialized:")
        print(f"  wn={wn} rad/s, zeta={zeta}, ki_scale={ki_scale}")
        print(f"  Kp diag: {np.diag(self.Kp).round(4)}")
        print(f"  Ki diag: {np.diag(self.Ki).round(6)}")
        print(f"  Kd diag: {np.diag(self.Kd).round(4)}")
        print(f"  Anti-windup clamp: {tau_max} N*m")
        print(f"  Settling time ~ {4/(zeta*wn):.1f}s")

    def reset(self):
        """Reset integral accumulator — call when changing desired attitude."""
        self.integral = np.zeros(3)

    def __call__(self, q, omega, t):
        """
        Compute PID control torque.

        Parameters
        ----------
        q     : np.ndarray (4,)  — current attitude quaternion
        omega : np.ndarray (3,)  — current angular velocity (rad/s)
        t     : float            — current time (s)

        Returns
        -------
        tau : np.ndarray (3,)  — control torque in body frame (N*m)
        """
        _, err_vec, err_deg = compute_error_quaternion(q, self.q_desired)

        # Integrator deadband: only accumulate when close to target
        # Prevents integral windup during large slews — the integral
        # should only activate to kill the final residual error,
        # not charge up while the satellite is still swinging toward target
        if err_deg < 5.0:
            self.integral += err_vec * self.dt

        # Anti-windup: clamp integral torque magnitude
        tau_integral = self.Ki @ self.integral
        tau_int_mag  = np.linalg.norm(tau_integral)
        if tau_int_mag > self.tau_max * 0.5:
            self.integral *= (self.tau_max * 0.5) / tau_int_mag

        tau_p = self.Kp @ err_vec        # proportional — pushes toward target
        tau_i = self.Ki @ self.integral  # integral     — kills steady-state error
        tau_d = self.Kd @ omega          # derivative   — damps spin

        return tau_p + tau_i - tau_d

    def attitude_error_deg(self, q):
        """Scalar attitude error in degrees — useful for logging."""
        _, _, err_deg = compute_error_quaternion(q, self.q_desired)
        return err_deg


# ─────────────────────────────────────────────
#  Quick Tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from simulator import Simulator, SatelliteConfig
    from disturbances import (
        DisturbanceStack, GravityGradientTorque,
        SolarRadiationTorque, MagneticDisturbanceTorque,
        AerodynamicTorque,
    )

    cfg = SatelliteConfig(mass=10.0, lx=0.6, ly=0.2, lz=0.2)
    ALT = 500.0
    DT  = 0.05

    def make_dist():
        return DisturbanceStack(
            GravityGradientTorque(I=cfg.I,     altitude_km=ALT),
            SolarRadiationTorque(area=50.0,
                                 cop_offset=np.array([0.0, 0.0, 0.10]),
                                 altitude_km=ALT),
            MagneticDisturbanceTorque(         altitude_km=ALT),
            AerodynamicTorque(area=50.0,
                              cop_offset=np.array([0.0, 0.0, 0.03]),
                              altitude_km=ALT),
        )

    for label, deg in [("45°", 45.0), ("90°", 90.0), ("180°", 180.0)]:
        print("=" * 50)
        print(f"Test: {label} initial error")
        print("=" * 50)
        angle = np.radians(deg)
        q0    = np.array([np.cos(angle/2), 0.0, np.sin(angle/2), 0.0])
        ctrl  = PIDController(I=cfg.I, wn=0.3, zeta=0.7,
                              ki_scale=0.02, tau_max=0.005, dt=DT)
        print()
        sim = Simulator(cfg, q0=q0, omega0=np.zeros(3), dt=DT, t_end=300.0,
                        controller=ctrl, disturbance=make_dist())
        r = sim.run()
        print(f"Final error: {ctrl.attitude_error_deg(r.q[-1]):.3f} deg\n")