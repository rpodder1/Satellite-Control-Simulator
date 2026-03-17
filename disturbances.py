"""
disturbances.py

Environmental disturbance torques acting on a spacecraft in LEO.

Four disturbances are modeled, in order of implementation complexity:

    1. Gravity gradient  — differential gravity across the satellite body
    2. Solar radiation   — photon pressure on reflective surfaces (dominant for large mirror satellites)
    3. Magnetic          — residual dipole interacting with Earth's field
    4. Aerodynamic       — atmospheric drag torque at low altitudes

Each disturbance is a callable class with signature:
    tau = disturbance(q, omega, t)  →  np.ndarray (3,)  N·m in body frame

A DisturbanceStack combines multiple disturbances into one callable,
which plugs directly into Simulator as the disturbance= argument.

Units throughout:
    distances  — meters
    angles     — radians
    time       — seconds
    torque     — N·m
"""

import numpy as np
from rigid_body import quat_to_rotation_matrix


# ─────────────────────────────────────────────
#  Physical Constants
# ─────────────────────────────────────────────

MU_EARTH   = 3.986004418e14   # m³/s²  — Earth gravitational parameter
RE_EARTH   = 6.371e6          # m      — Earth mean radius
SOLAR_FLUX = 1361.0           # W/m²   — solar irradiance at 1 AU
C_LIGHT    = 2.998e8          # m/s    — speed of light
B0_EARTH   = 3.12e-5          # T      — Earth magnetic dipole strength at equator


# ─────────────────────────────────────────────
#  Orbital Utilities
# ─────────────────────────────────────────────

def orbital_radius(altitude_km):
    """Convert altitude in km to orbital radius in meters."""
    return RE_EARTH + altitude_km * 1e3


def orbital_velocity(altitude_km):
    """Circular orbital velocity at given altitude (m/s)."""
    r = orbital_radius(altitude_km)
    return np.sqrt(MU_EARTH / r)


def sun_vector_inertial(t):
    """
    Approximate unit vector from Earth to Sun in inertial frame.
    Models Earth's orbit as circular, one revolution per year.

    Parameters
    ----------
    t : float  — simulation time (s)

    Returns
    -------
    s_hat : np.ndarray (3,)  — unit vector toward Sun, inertial frame
    """
    omega_earth = 2 * np.pi / (365.25 * 24 * 3600)   # rad/s — Earth orbital rate
    angle = omega_earth * t
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def nadir_vector_inertial(t, altitude_km):
    """
    Approximate unit vector from satellite toward Earth center (nadir),
    in inertial frame. Models circular orbit in equatorial plane.

    Parameters
    ----------
    t           : float  — simulation time (s)
    altitude_km : float  — orbital altitude

    Returns
    -------
    nadir_hat : np.ndarray (3,)  — unit nadir vector, inertial frame
    """
    r = orbital_radius(altitude_km)
    v = orbital_velocity(altitude_km)
    omega_orbit = v / r                             # orbital angular rate (rad/s)
    angle = omega_orbit * t
    # Satellite position in inertial frame
    pos = np.array([np.cos(angle), np.sin(angle), 0.0])
    return -pos                                     # nadir points toward Earth center


def magnetic_field_body(q, t, altitude_km):
    """
    Earth's magnetic field at satellite location, expressed in body frame.
    Uses a simplified tilted dipole model.

    Parameters
    ----------
    q           : np.ndarray (4,)  — current attitude quaternion
    t           : float            — simulation time (s)
    altitude_km : float            — orbital altitude

    Returns
    -------
    B_body : np.ndarray (3,)  — magnetic field vector in body frame (T)
    """
    r = orbital_radius(altitude_km)
    omega_orbit = orbital_velocity(altitude_km) / r
    angle = omega_orbit * t

    # Dipole model in inertial frame (simplified — aligned with Earth's spin axis)
    B_mag = B0_EARTH * (RE_EARTH / r)**3
    # Field varies as satellite moves through orbit
    B_inertial = B_mag * np.array([
        2 * np.sin(angle) * np.cos(np.radians(20)),   # slight tilt
        np.cos(angle),
        np.sin(np.radians(20))
    ])

    # Rotate to body frame: B_body = R^T * B_inertial
    R = quat_to_rotation_matrix(q)
    return R.T @ B_inertial


# ─────────────────────────────────────────────
#  Disturbance 1: Gravity Gradient
# ─────────────────────────────────────────────

class GravityGradientTorque:
    """
    Gravity gradient torque — differential gravitational pull across the body.

    Tries to align the satellite's minimum inertia axis with the local vertical.
    Useful for passive stabilization of elongated satellites.

    Torque magnitude: τ ~ (3μ/2r³) · |Imax - Imin| · sin(2θ)
    where θ is misalignment angle from nadir.

    Parameters
    ----------
    I           : np.ndarray (3,3)  — inertia tensor (kg·m²)
    altitude_km : float             — orbital altitude (km)
    """

    def __init__(self, I, altitude_km=500.0):
        self.I           = I
        self.altitude_km = altitude_km
        r = orbital_radius(altitude_km)
        self.coeff = 3.0 * MU_EARTH / (2.0 * r**3)   # precomputed, constant for circular orbit
        print(f"GravityGradientTorque: altitude={altitude_km}km, "
              f"coeff={self.coeff:.4e} rad/s²")

    def __call__(self, q, omega, t):
        """
        Compute gravity gradient torque in body frame.

        Parameters
        ----------
        q     : np.ndarray (4,)  — current attitude quaternion
        omega : np.ndarray (3,)  — angular velocity (unused here)
        t     : float            — current time (s)

        Returns
        -------
        tau : np.ndarray (3,)  — torque in body frame (N·m)
        """
        # Nadir unit vector in inertial frame
        nadir_inertial = nadir_vector_inertial(t, self.altitude_km)

        # Rotate nadir into body frame
        R = quat_to_rotation_matrix(q)
        nadir_body = R.T @ nadir_inertial   # body = R^T * inertial

        # Gravity gradient torque: τ = (3μ/r³) * nadir × (I · nadir)
        I_nadir = self.I @ nadir_body
        tau = self.coeff * np.cross(nadir_body, I_nadir)

        return tau


# ─────────────────────────────────────────────
#  Disturbance 2: Solar Radiation Pressure
# ─────────────────────────────────────────────

class SolarRadiationTorque:
    """
    Solar radiation pressure torque — photon momentum transfer on surfaces.

    THIS IS THE DOMINANT DISTURBANCE FOR REFLECT ORBITAL.
    Their large mirror means SRP torque can be 10-100x larger than
    other disturbances, requiring continuous active compensation.

    Model assumes a flat mirror surface. Torque arises from offset
    between center of mass (CoM) and center of pressure (CoP).

    τ_srp = (Φ/c) · Cr · A · (r_cop × n̂) · max(0, n̂ · s_hat)

    Parameters
    ----------
    area        : float            — mirror area (m²)
    cop_offset  : np.ndarray (3,)  — CoP position relative to CoM in body frame (m)
    Cr          : float            — radiation pressure coefficient
                                     1.0 = full absorption, 2.0 = perfect reflection
    altitude_km : float            — orbital altitude (km) for eclipse detection
    """

    def __init__(self, area=1.0, cop_offset=None, Cr=1.8, altitude_km=500.0):
        self.area        = area
        self.Cr          = Cr
        self.altitude_km = altitude_km
        self.pressure    = (SOLAR_FLUX / C_LIGHT) * Cr   # N/m²

        # Default CoP offset: 5cm along body Z (mirror above the bus)
        self.cop_offset = (
            cop_offset if cop_offset is not None
            else np.array([0.0, 0.0, 0.05])
        )

        # Mirror normal in body frame — points along body Z by default
        self.mirror_normal_body = np.array([0.0, 0.0, 1.0])

        print(f"SolarRadiationTorque: area={area}m², Cr={Cr}, "
              f"CoP offset={self.cop_offset}m")
        print(f"  Peak torque ≈ {self.pressure * area * np.linalg.norm(self.cop_offset):.4e} N·m")

    def __call__(self, q, omega, t):
        """
        Compute SRP torque in body frame.

        Parameters
        ----------
        q     : np.ndarray (4,)  — current attitude quaternion
        omega : np.ndarray (3,)  — angular velocity (unused)
        t     : float            — current time (s)

        Returns
        -------
        tau : np.ndarray (3,)  — torque in body frame (N·m)
        """
        # Sun vector in inertial frame
        sun_inertial = sun_vector_inertial(t)

        # Rotate sun vector into body frame
        R = quat_to_rotation_matrix(q)
        sun_body = R.T @ sun_inertial

        # Mirror normal in body frame
        n_hat = self.mirror_normal_body

        # Illumination factor — zero if sun is behind the mirror
        # (dot product of mirror normal with sun direction)
        illumination = max(0.0, np.dot(n_hat, sun_body))

        if illumination < 1e-6:
            return np.zeros(3)   # satellite in eclipse or mirror facing away

        # SRP force vector in body frame
        # F = pressure * area * illumination * sun_body_direction
        F_srp = self.pressure * self.area * illumination * sun_body

        # Torque = CoP_offset × F_srp
        tau = np.cross(self.cop_offset, F_srp)

        return tau


# ─────────────────────────────────────────────
#  Disturbance 3: Magnetic Torque
# ─────────────────────────────────────────────

class MagneticDisturbanceTorque:
    """
    Residual magnetic dipole torque — interaction between satellite's
    residual magnetic moment and Earth's field.

    τ_mag = m × B

    where m is the satellite's residual dipole moment (from electronics,
    current loops, magnetized components) and B is Earth's field.

    Note: magnetorquers work on the same principle but with a *commanded*
    dipole. This models the *unwanted* residual dipole.

    Parameters
    ----------
    residual_dipole : np.ndarray (3,)  — residual magnetic dipole (A·m²)
                                         typical small satellite: 0.01–0.1 A·m²
    altitude_km     : float            — orbital altitude (km)
    """

    def __init__(self, residual_dipole=None, altitude_km=500.0):
        self.altitude_km = altitude_km
        self.m = (
            residual_dipole if residual_dipole is not None
            else np.array([0.01, 0.005, 0.008])   # A·m² — typical small satellite
        )
        print(f"MagneticDisturbanceTorque: dipole={self.m} A·m², "
              f"|m|={np.linalg.norm(self.m):.4f} A·m²")

    def __call__(self, q, omega, t):
        """
        Compute magnetic disturbance torque in body frame.
        """
        B_body = magnetic_field_body(q, t, self.altitude_km)
        tau = np.cross(self.m, B_body)
        return tau


# ─────────────────────────────────────────────
#  Disturbance 4: Aerodynamic Torque
# ─────────────────────────────────────────────

class AerodynamicTorque:
    """
    Aerodynamic drag torque from residual atmosphere in LEO.

    Significant below ~600km. Dominant below ~400km.
    Negligible above ~700km.

    τ_aero = 0.5 * ρ * v² * Cd * A * (r_cp × v̂)

    Parameters
    ----------
    area        : float            — cross-sectional area (m²)
    cop_offset  : np.ndarray (3,)  — CoP offset from CoM in body frame (m)
    Cd          : float            — drag coefficient (~2.2 for flat plate)
    altitude_km : float            — orbital altitude (km)
    """

    # Approximate atmospheric density model (kg/m³) at various altitudes
    # Based on US Standard Atmosphere 1976
    _ALT_KM  = np.array([200, 300, 400, 500, 600, 700, 800])
    _DENSITY = np.array([2.54e-10, 1.92e-11, 2.80e-12, 5.22e-13,
                         1.14e-13, 3.07e-14, 1.14e-14])

    def __init__(self, area=1.0, cop_offset=None, Cd=2.2, altitude_km=500.0):
        self.area        = area
        self.Cd          = Cd
        self.altitude_km = altitude_km

        # Interpolate atmospheric density at this altitude
        self.rho = float(np.interp(altitude_km, self._ALT_KM, self._DENSITY))
        self.v   = orbital_velocity(altitude_km)

        self.cop_offset = (
            cop_offset if cop_offset is not None
            else np.array([0.0, 0.0, 0.03])   # 3cm offset
        )

        # Velocity direction in inertial frame — along-track (Y for equatorial orbit)
        # This is approximate; a full model would use the actual velocity vector
        self.drag_coeff = 0.5 * self.rho * self.v**2 * Cd * area

        print(f"AerodynamicTorque: altitude={altitude_km}km, "
              f"ρ={self.rho:.3e} kg/m³, v={self.v:.1f} m/s")
        print(f"  Peak torque ≈ {self.drag_coeff * np.linalg.norm(self.cop_offset):.4e} N·m")

    def __call__(self, q, omega, t):
        """
        Compute aerodynamic torque in body frame.
        """
        # Velocity direction in inertial frame (along-track, equatorial orbit)
        omega_orbit = self.v / orbital_radius(self.altitude_km)
        angle = omega_orbit * t
        v_hat_inertial = np.array([-np.sin(angle), np.cos(angle), 0.0])

        # Rotate velocity direction into body frame
        R = quat_to_rotation_matrix(q)
        v_hat_body = R.T @ v_hat_inertial

        # Drag force opposes velocity
        F_aero = -self.drag_coeff * v_hat_body

        # Torque = CoP_offset × F_drag
        tau = np.cross(self.cop_offset, F_aero)

        return tau


# ─────────────────────────────────────────────
#  Disturbance Stack
# ─────────────────────────────────────────────

class DisturbanceStack:
    """
    Combines multiple disturbance models into a single callable.
    Plugs directly into Simulator as the disturbance= argument.

    Parameters
    ----------
    *disturbances : any number of disturbance callables

    Example
    -------
    dist = DisturbanceStack(
        GravityGradientTorque(I=cfg.I, altitude_km=500),
        SolarRadiationTorque(area=50.0, cop_offset=np.array([0,0,0.1])),
    )
    sim = Simulator(config=cfg, controller=ctrl, disturbance=dist)
    """

    def __init__(self, *disturbances):
        self.disturbances = disturbances
        print(f"DisturbanceStack: {len(disturbances)} disturbance(s) active")

    def __call__(self, q, omega, t):
        total = np.zeros(3)
        for d in self.disturbances:
            total += d(q, omega, t)
        return total


# ─────────────────────────────────────────────
#  Quick Tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from simulator import Simulator, SatelliteConfig
    from pid_controller import PIDController

    cfg = SatelliteConfig(mass=10.0, lx=0.6, ly=0.2, lz=0.2)
    ALT = 500.0
    DT  = 0.05

    angle = np.radians(45.0)
    q0    = np.array([np.cos(angle/2), 0.0, np.sin(angle/2), 0.0])

    # Check each disturbance prints correctly and runs without error
    print("Instantiating disturbances...")
    gg   = GravityGradientTorque(I=cfg.I, altitude_km=ALT)
    srp  = SolarRadiationTorque(area=50.0, cop_offset=np.array([0.0,0.0,0.10]),
                                altitude_km=ALT)
    mag  = MagneticDisturbanceTorque(altitude_km=ALT)
    aero = AerodynamicTorque(area=50.0, cop_offset=np.array([0.0,0.0,0.03]),
                             altitude_km=ALT)
    stack = DisturbanceStack(gg, srp, mag, aero)
    print()

    # Sample torques at t=0, identity attitude
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    w_z  = np.zeros(3)
    print("Torque samples at t=0, identity attitude:")
    print(f"  Gravity gradient: {gg(q_id, w_z, 0) * 1e6} uN·m")
    print(f"  SRP:              {srp(q_id, w_z, 0) * 1e6} uN·m")
    print(f"  Magnetic:         {mag(q_id, w_z, 0) * 1e6} uN·m")
    print(f"  Aerodynamic:      {aero(q_id, w_z, 0) * 1e6} uN·m")
    print(f"  Total stack:      {stack(q_id, w_z, 0) * 1e6} uN·m")
    print()

    # Run a quick simulation to confirm stack integrates cleanly
    print("Running 60s simulation with all disturbances + PID controller...")
    ctrl = PIDController(I=cfg.I, wn=0.3, zeta=0.7,
                         ki_scale=0.02, tau_max=0.005, dt=DT)
    sim  = Simulator(cfg, q0=q0, omega0=np.zeros(3), dt=DT, t_end=60.0,
                     controller=ctrl, disturbance=stack)
    r    = sim.run()
    print(f"Final error after 60s: {ctrl.attitude_error_deg(r.q[-1]):.3f} deg")