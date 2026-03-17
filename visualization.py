"""
visualization.py

Portfolio-quality visualizations for the attitude control simulator.

Produces three outputs:
    1. animated_attitude.gif  — 3D satellite animation from tumble to settled
    2. summary_plot.png       — polished 4-panel diagnostic for README
    3. reflection_geometry.png — sunlight reflection mission context:
                                 when/where a LEO satellite can reflect
                                 sunlight to a ground target

Usage
-----
    python visualization.py

All outputs saved to current directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec

from rigid_body import build_inertia_tensor, quat_to_rotation_matrix
from simulator import Simulator, SatelliteConfig
from pid_controller import PIDController
from disturbances import (
    DisturbanceStack,
    GravityGradientTorque,
    SolarRadiationTorque,
    MagneticDisturbanceTorque,
    AerodynamicTorque,
    orbital_radius,
    orbital_velocity,
    sun_vector_inertial,
    nadir_vector_inertial,
    MU_EARTH,
    RE_EARTH,
)


# ─────────────────────────────────────────────
#  Shared simulation — run once, use everywhere
# ─────────────────────────────────────────────

def run_showcase_simulation():
    """
    Run the main simulation used by all three visualizations.
    45° initial error, PID controller, all disturbances, 200s.

    Desired attitude: mirror normal bisects sun–target angle
    (specular reflection condition for sunlight redirection mission).
    """
    print("Running showcase simulation...")

    # Realistic satellite: large deployable reflector bus
    # 150kg, 2m × 1m × 0.5m body, 50m² mirror
    cfg = SatelliteConfig(mass=150.0, lx=2.0, ly=1.0, lz=0.5)
    ALT = 500.0

    # Inertia at this scale: Izz ≈ 21 kg·m², ~600× larger than 10kg CubeSat
    # SRP torque also scales up: 30cm CoP offset on large deployable
    dist = DisturbanceStack(
        GravityGradientTorque(I=cfg.I,      altitude_km=ALT),
        SolarRadiationTorque(area=50.0,
                             cop_offset=np.array([0.0, 0.0, 0.30]),
                             altitude_km=ALT),
        MagneticDisturbanceTorque(          altitude_km=ALT),
        AerodynamicTorque(area=50.0,
                          cop_offset=np.array([0.0, 0.0, 0.15]),
                          altitude_km=ALT),
    )

    # ── Compute desired attitude from reflection geometry ─────
    # Sun vector in inertial frame (at t=0)
    sun_hat = sun_vector_inertial(0.0)

    # ── Ground target: Mammoth Solar Farm, Starke County, Indiana
    # Largest solar farm in the US midwest — 41.3°N, 86.6°W
    target_lat = np.radians(41.3)
    target_lon = np.radians(-86.6)
    target_ecef = np.array([
        np.cos(target_lat) * np.cos(target_lon),
        np.cos(target_lat) * np.sin(target_lon),
        np.sin(target_lat),
    ])

    # Satellite position at t=0 (equatorial orbit, 0° longitude)
    r = orbital_radius(ALT)
    sat_pos = np.array([1.0, 0.0, 0.0])   # unit vector
    to_target = target_ecef - sat_pos
    target_hat = to_target / np.linalg.norm(to_target)

    # Mirror normal = bisector of sun and target directions
    # (specular reflection: normal bisects incoming and outgoing rays)
    mirror_normal = sun_hat + target_hat
    mirror_normal /= np.linalg.norm(mirror_normal)

    # Mirror is body Z — build rotation matrix that maps body Z → mirror_normal
    body_z    = np.array([0.0, 0.0, 1.0])
    axis      = np.cross(body_z, mirror_normal)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        q_desired = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis  /= axis_norm
        angle  = np.arccos(np.clip(np.dot(body_z, mirror_normal), -1, 1))
        q_desired = np.array([
            np.cos(angle/2),
            axis[0]*np.sin(angle/2),
            axis[1]*np.sin(angle/2),
            axis[2]*np.sin(angle/2),
        ])

    print(f"Target: Mammoth Solar Farm, Indiana (41.3°N, 86.6°W)")
    print(f"Sun vector:           {np.round(sun_hat, 3)}")
    print(f"Mirror normal target: {np.round(mirror_normal, 3)}")
    print(f"Desired quaternion:   {np.round(q_desired, 4)}")

    ctrl = PIDController(I=cfg.I, wn=0.5, zeta=0.7,
                         ki_scale=0.03, tau_max=0.5, dt=0.05,
                         q_desired=q_desired)

    # ── Initial attitude: 45° away from desired ───────────────
    # Rotate q_desired by 45° about body Y to get starting attitude
    perturb = np.array([np.cos(np.radians(45)/2), 0.0,
                        np.sin(np.radians(45)/2), 0.0])
    from rigid_body import quat_multiply
    q0 = quat_multiply(perturb, q_desired)

    sim = Simulator(
        config      = cfg,
        q0          = q0,
        omega0      = np.array([0.02, 0.05, -0.03]),
        t_end       = 300.0,
        dt          = 0.05,
        controller  = ctrl,
        disturbance = dist,
    )

    results = sim.run()
    print(f"Final pointing error: {ctrl.attitude_error_deg(results.q[-1]):.3f}°\n")
    return results, cfg, ctrl


# ─────────────────────────────────────────────
#  Satellite geometry helper
# ─────────────────────────────────────────────

def satellite_faces(R, scale=1.0):
    """
    Return the faces of a satellite box + solar panels + mirror.

    Mirror is a large flat panel on the +Z face (body Z = mirror normal).
    Solar panels extend along Y axis.

    Returns
    -------
    body_faces   : list of (4,3) arrays
    panel_faces  : list of (4,3) arrays — solar panels
    mirror_faces : list of (4,3) arrays — reflective mirror
    """
    sx, sy, sz = 0.6*scale, 0.12*scale, 0.12*scale

    # Body vertices
    bv = np.array([
        [-sx, -sy, -sz], [ sx, -sy, -sz], [ sx,  sy, -sz], [-sx,  sy, -sz],
        [-sx, -sy,  sz], [ sx, -sy,  sz], [ sx,  sy,  sz], [-sx,  sy,  sz],
    ])
    bv = (R @ bv.T).T

    body_faces = [
        [bv[0], bv[1], bv[2], bv[3]],
        [bv[4], bv[5], bv[6], bv[7]],
        [bv[0], bv[1], bv[5], bv[4]],
        [bv[2], bv[3], bv[7], bv[6]],
        [bv[0], bv[3], bv[7], bv[4]],
        [bv[1], bv[2], bv[6], bv[5]],
    ]

    # Solar panels — thin, extend along Y
    pw = 0.04*scale
    pv = np.array([
        [ sx,  sy,       sz-pw], [-sx,  sy,       sz-pw],
        [-sx,  sy*4.5,   sz-pw], [ sx,  sy*4.5,   sz-pw],
        [ sx, -sy,       sz-pw], [-sx, -sy,       sz-pw],
        [-sx, -sy*4.5,   sz-pw], [ sx, -sy*4.5,   sz-pw],
    ])
    pv = (R @ pv.T).T
    panel_faces = [
        [pv[0], pv[1], pv[2], pv[3]],
        [pv[4], pv[5], pv[6], pv[7]],
    ]

    # Mirror — large flat panel proud of +Z face, this is what reflects sunlight
    mw = sx * 1.1    # slightly wider than body
    mh = sy * 1.1
    mz = sz + 0.02*scale   # sits just above +Z face
    mv = np.array([
        [-mw, -mh, mz], [ mw, -mh, mz],
        [ mw,  mh, mz], [-mw,  mh, mz],
    ])
    mv = (R @ mv.T).T
    mirror_faces = [[mv[0], mv[1], mv[2], mv[3]]]

    return body_faces, panel_faces, mirror_faces


# ─────────────────────────────────────────────
#  1. Animated GIF
# ─────────────────────────────────────────────

def make_animation(results, ctrl, filename='animated_attitude.gif', n_frames=160):
    """
    Cinematic 3D satellite attitude control animation.

    Features:
      - Deep space starfield background
      - Slowly rotating camera orbit
      - Satellite with per-face shading based on sun angle
      - Glowing body-frame axes with alpha trails
      - Ghost wireframe showing target attitude
      - Angular velocity vector with fading trail
      - HUD overlay: error bar, phase label, time/error readout
      - Camera locks on as satellite settles

    Parameters
    ----------
    results  : SimulationResults
    filename : str  — output filename
    n_frames : int  — animation frames (160 = ~8s at 20fps)
    """
    from rigid_body import quat_multiply, quat_conjugate
    import matplotlib.patheffects as pe

    print(f"Rendering animation ({n_frames} frames)...")

    total_steps = len(results.t)
    indices     = np.linspace(0, total_steps - 1, n_frames, dtype=int)
    t_frames    = results.t[indices]
    q_frames    = results.q[indices]
    omega_frames = results.omega[indices]

    # Precompute error for every frame using actual desired attitude
    q_des = ctrl.q_desired
    err_frames = []
    for q in q_frames:
        qe = quat_multiply(q_des, quat_conjugate(q))
        if qe[0] < 0: qe = -qe
        err_frames.append(np.degrees(2.0 * np.arccos(np.clip(abs(qe[0]), 0.0, 1.0))))
    err_frames = np.array(err_frames)
    max_err    = err_frames[0]

    # Starfield — fixed random positions
    rng         = np.random.default_rng(42)
    n_stars     = 280
    star_az     = rng.uniform(0, 2*np.pi, n_stars)
    star_el     = rng.uniform(-np.pi/2, np.pi/2, n_stars)
    star_r      = rng.uniform(2.8, 3.5, n_stars)
    star_sizes  = rng.uniform(0.4, 2.2, n_stars)
    star_alpha  = rng.uniform(0.3, 1.0, n_stars)
    sx = star_r * np.cos(star_el) * np.cos(star_az)
    sy = star_r * np.cos(star_el) * np.sin(star_az)
    sz = star_r * np.sin(star_el)

    # Ghost target attitude (identity = R = I)
    R_target = np.eye(3)

    BODY_COLORS = ['#ff5555', '#44dd66', '#5599ff']
    BG          = '#05050f'

    fig = plt.figure(figsize=(9, 7.2), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(BG)

    def style_ax():
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_zlim(-1.4, 1.4)
        ax.set_box_aspect([1, 1, 1])
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#0a0a1a')
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    def face_brightness(normal_world, sun_dir):
        """Simple Lambertian shading — brighter faces facing the sun."""
        dot = np.dot(normal_world, sun_dir)
        return np.clip(dot, 0.08, 1.0)

    def draw_frame(fi):
        ax.cla()
        style_ax()

        q      = q_frames[fi]
        t      = t_frames[fi]
        omega  = omega_frames[fi]
        err    = err_frames[fi]
        R      = quat_to_rotation_matrix(q)
        sun    = sun_vector_inertial(t)
        mirror_normal_world = R @ np.array([0.0, 0.0, 1.0])

        # ── Camera: slow subtle drift + slight rise as satellite settles ──
        settle_frac = 1.0 - min(1.0, err / max_err)
        azim = 32                                  # fixed camera angle
        elev = 20 + settle_frac * 6            # barely noticeable rise
        ax.view_init(elev=elev, azim=azim)

        # ── Starfield ─────────────────────────────────────────
        ax.scatter(sx, sy, sz, s=star_sizes, c='white',
                   alpha=star_alpha * 0.7, zorder=0, depthshade=False)

        # ── Sun ray + reflection geometry ─────────────────────
        sun_tip = sun * 1.25
        ax.quiver(sun_tip[0]-sun[0]*0.35,
                  sun_tip[1]-sun[1]*0.35,
                  sun_tip[2]-sun[2]*0.35,
                  sun[0]*0.3, sun[1]*0.3, sun[2]*0.3,
                  color='#ffe566', linewidth=1.2, alpha=0.55,
                  arrow_length_ratio=0.35)
        ax.text(sun_tip[0], sun_tip[1], sun_tip[2],
                '☀', color='#ffe566', fontsize=9, alpha=0.6,
                ha='center', va='center')

        # Mirror normal (body Z) and reflected beam        mirror_normal_world = R @ np.array([0.0, 0.0, 1.0])
        # Reflected ray: r = d - 2(d·n)n where d = incident direction
        d_inc = -sun
        ref   = d_inc - 2 * np.dot(d_inc, mirror_normal_world) * mirror_normal_world
        ref_hat = ref / (np.linalg.norm(ref) + 1e-9)
        # Only draw reflected beam when it points toward Earth (downward)
        if ref_hat[2] < 0:
            ax.quiver(0, 0, 0,
                      ref_hat[0]*0.75, ref_hat[1]*0.75, ref_hat[2]*0.75,
                      color='#ff8822', linewidth=1.8, alpha=0.65,
                      arrow_length_ratio=0.12)
            ax.text(ref_hat[0]*0.82, ref_hat[1]*0.82, ref_hat[2]*0.82,
                    'reflected', color='#ff8822', fontsize=7, alpha=0.55)

        # ── Satellite with Lambertian shading ─────────────────
        sx_s  = 0.60*0.82; sy_s = 0.12*0.82; sz_s = 0.12*0.82
        verts_body = np.array([
            [-sx_s,-sy_s,-sz_s],[sx_s,-sy_s,-sz_s],[sx_s,sy_s,-sz_s],[-sx_s,sy_s,-sz_s],
            [-sx_s,-sy_s, sz_s],[sx_s,-sy_s, sz_s],[sx_s,sy_s, sz_s],[-sx_s,sy_s, sz_s],
        ])
        verts_body = (R @ verts_body.T).T

        face_defs = [
            ([0,1,2,3], np.array([0,0,-1])),   # -Z face
            ([4,5,6,7], np.array([0,0, 1])),   # +Z face
            ([0,1,5,4], np.array([0,-1,0])),   # -Y face
            ([2,3,7,6], np.array([0, 1,0])),   # +Y face
            ([0,3,7,4], np.array([-1,0,0])),   # -X face
            ([1,2,6,5], np.array([ 1,0,0])),   # +X face
        ]

        # Sort faces back-to-front for painter's algorithm
        cam_dir = np.array([np.cos(np.radians(elev))*np.cos(np.radians(azim)),
                            np.cos(np.radians(elev))*np.sin(np.radians(azim)),
                            np.sin(np.radians(elev))])
        face_depths = []
        for idxs, n_local in face_defs:
            center = verts_body[idxs].mean(axis=0)
            face_depths.append(np.dot(center, cam_dir))
        sorted_faces = sorted(zip(face_depths, face_defs), key=lambda x: x[0])

        for _, (idxs, n_local) in sorted_faces:
            n_world  = R @ n_local
            bright   = face_brightness(n_world, sun)
            # Blue-grey body, brightened on sun-facing side
            r_c = 0.12 + bright * 0.22
            g_c = 0.18 + bright * 0.28
            b_c = 0.32 + bright * 0.38
            face_verts = [verts_body[i] for i in idxs]
            fc = Poly3DCollection([face_verts], alpha=0.92)
            fc.set_facecolor((r_c, g_c, b_c))
            fc.set_edgecolor((r_c*0.6, g_c*0.6, b_c*0.9))
            fc.set_linewidth(0.4)
            ax.add_collection3d(fc)

        # Solar panels — thin, deep blue, along +Y and -Y
        pw = 0.04*0.82
        for sign in [1, -1]:
            pv = np.array([
                [ sx_s,  sign*sy_s,       sz_s-pw],
                [-sx_s,  sign*sy_s,       sz_s-pw],
                [-sx_s,  sign*sy_s*4.5,   sz_s-pw],
                [ sx_s,  sign*sy_s*4.5,   sz_s-pw],
            ])
            pv = (R @ pv.T).T
            n_panel  = R @ np.array([0, 0, 1])
            bright_p = face_brightness(n_panel, sun)
            panel_col = Poly3DCollection([pv], alpha=0.92)
            panel_col.set_facecolor((0.04, 0.10 + bright_p*0.12,
                                     0.26 + bright_p*0.18))
            panel_col.set_edgecolor('#2255aa')
            panel_col.set_linewidth(0.5)
            ax.add_collection3d(panel_col)

        # Mirror — large flat panel on +Z face, bright silver/white
        mw = sx_s * 1.1; mh = sy_s * 1.1; mz = sz_s + 0.02*0.82
        mv = np.array([
            [-mw, -mh, mz], [ mw, -mh, mz],
            [ mw,  mh, mz], [-mw,  mh, mz],
        ])
        mv = (R @ mv.T).T
        mirror_bright = face_brightness(mirror_normal_world, sun)
        # Mirror is bright silver when facing sun, dark when away
        mc = 0.55 + mirror_bright * 0.45
        mirror_col = Poly3DCollection([mv], alpha=0.95)
        mirror_col.set_facecolor((mc, mc, mc * 0.92))   # silver tint
        mirror_col.set_edgecolor('#aaaacc')
        mirror_col.set_linewidth(0.8)
        ax.add_collection3d(mirror_col)
        # Label the mirror face
        mirror_center = mv.mean(axis=0)
        ax.text(mirror_center[0], mirror_center[1], mirror_center[2] + 0.06,
                'MIRROR', color='#ccccff', fontsize=6, alpha=0.7,
                ha='center', va='center')

        # ── Angular velocity trail ────────────────────────────
        trail_len = min(fi + 1, 18)
        for k in range(trail_len - 1):
            ki     = max(0, fi - trail_len + k + 1)
            w_tip  = omega_frames[ki] * 1.8
            alpha  = (k / trail_len) * 0.5
            ax.plot([0, w_tip[0]], [0, w_tip[1]], [0, w_tip[2]],
                    color='#ffaa22', linewidth=1.0, alpha=alpha)
        w_tip = omega * 1.8
        ax.quiver(0, 0, 0, w_tip[0], w_tip[1], w_tip[2],
                  color='#ffcc44', linewidth=2.0, alpha=0.85,
                  arrow_length_ratio=0.22)

        # ── Body frame axes with glow ─────────────────────────
        ax_len = 0.95
        for i in range(3):
            d = R[:, i] * ax_len
            # Outer glow (thick, transparent)
            ax.plot([0, d[0]], [0, d[1]], [0, d[2]],
                    color=BODY_COLORS[i], linewidth=5, alpha=0.12)
            # Inner line
            ax.quiver(0, 0, 0, d[0], d[1], d[2],
                      color=BODY_COLORS[i], linewidth=2.2,
                      alpha=0.95, arrow_length_ratio=0.16)
            ax.text(d[0]*1.06, d[1]*1.06, d[2]*1.06,
                    ['X', 'Y', 'Z'][i],
                    color=BODY_COLORS[i], fontsize=9,
                    fontweight='bold', alpha=0.9,
                    ha='center', va='center')

        # ── Inertial frame (faint) ────────────────────────────
        for i in range(3):
            d = np.eye(3)[i] * 0.85
            ax.plot([0, d[0]], [0, d[1]], [0, d[2]],
                    color=BODY_COLORS[i], alpha=0.1,
                    linewidth=0.8, linestyle='--')

        # ── HUD: error bar as text-based vertical strip ───────
        err_frac   = min(1.0, err / max_err)
        if err_frac > 0.5:
            hud_r = 1.0;   hud_g = (1 - err_frac) * 2 * 0.8;  hud_b = 0.1
        else:
            hud_r = err_frac * 2;  hud_g = 0.85;  hud_b = 0.2
        hud_color = (hud_r, hud_g, hud_b)

        # Phase label
        if err > 5.0:
            phase = 'SLEWING'
            phase_col = '#ff6644'
        elif err > 1.0:
            phase = 'CONVERGING'
            phase_col = '#ffcc44'
        else:
            phase = 'SETTLED ✓'
            phase_col = '#44ee88'

        # ── Text HUD ─────────────────────────────────────────
        fig.texts.clear()

        # Title
        fig.text(0.5, 0.965, 'Attitude Control Simulation',
                 ha='center', color='#aaaacc', fontsize=11,
                 fontweight='normal', alpha=0.9)

        # Time and error
        fig.text(0.04, 0.92, f't = {t:6.1f} s',
                 color='#8888aa', fontsize=9, family='monospace')
        fig.text(0.04, 0.87, f'err = {err:5.1f}°',
                 color=hud_color, fontsize=10,
                 family='monospace', fontweight='bold')

        # Phase
        fig.text(0.04, 0.82, phase,
                 color=phase_col, fontsize=9,
                 fontweight='bold', alpha=0.9)

        # Error bar — drawn as rectangle patches on figure canvas
        from matplotlib.patches import Rectangle
        bar_l = 0.025; bar_b = 0.18; bar_w = 0.012; bar_h = 0.55
        # Background
        fig.patches.clear()
        fig.add_artist(Rectangle((bar_l, bar_b), bar_w, bar_h,
                                  transform=fig.transFigure,
                                  facecolor='#111122',
                                  edgecolor='#2a2a4a',
                                  linewidth=0.5, zorder=3))
        # Fill — grows from bottom as error reduces
        fill_h = bar_h * (1.0 - err_frac)
        fill_b = bar_b + bar_h - fill_h
        fig.add_artist(Rectangle((bar_l, fill_b), bar_w, fill_h,
                                  transform=fig.transFigure,
                                  facecolor=hud_color,
                                  alpha=0.85, zorder=4))

        # Footer
        fig.text(0.5, 0.022,
                 '6-DOF Rigid Body Dynamics  ·  PID Controller  ·  '
                 'Target: Mammoth Solar Farm, Indiana',
                 ha='center', color='#444466', fontsize=7.5)

        return []

    ani = animation.FuncAnimation(
        fig, draw_frame,
        frames=n_frames,
        interval=50,
        blit=False,
    )

    ani.save(filename, writer='pillow', fps=20, dpi=110)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
#  2. Summary Plot
# ─────────────────────────────────────────────

def make_summary_plot(results, ctrl, filename='summary_plot.png'):
    """
    Polished 4-panel summary plot for README / portfolio.

    Parameters
    ----------
    results  : SimulationResults
    ctrl     : PIDController
    filename : str  — output filename
    """
    print("Rendering summary plot...")

    t         = results.t
    error_deg = results.attitude_error_deg(ctrl.q_desired)
    omega_deg = np.degrees(results.omega)
    tau_mnm   = results.tau_total * 1000   # N·m → mN·m

    fig = plt.figure(figsize=(12, 10), facecolor='#0d1117')
    gs  = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32,
                   left=0.09, right=0.97, top=0.91, bottom=0.07)

    DARK_BG    = '#0d1117'
    PANEL_BG   = '#161b22'
    GRID_COL   = '#21262d'
    TEXT_COL   = '#c9d1d9'
    MUTED_COL  = '#8b949e'
    COLS       = ['#ff6b6b', '#51cf66', '#74c0fc']
    AMBER      = '#ffd43b'
    ORANGE     = '#ff922b'

    def style_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=MUTED_COL, labelsize=9)
        ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight='normal', pad=8)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.8)
        ax.spines['bottom'].set_color(GRID_COL)
        ax.spines['top'].set_color(GRID_COL)
        ax.spines['left'].set_color(GRID_COL)
        ax.spines['right'].set_color(GRID_COL)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(MUTED_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.xaxis.label.set_color(TEXT_COL)

    # ── Panel 1: Attitude error ───────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])   # full-width top panel
    ax1.fill_between(t, error_deg, alpha=0.15, color=AMBER)
    ax1.plot(t, error_deg, color=AMBER, linewidth=1.5)
    ax1.axhline(1.0, color=MUTED_COL, linewidth=0.8, linestyle='--', alpha=0.5,
                label='1° threshold')
    ax1.set_ylabel('Attitude error (°)', fontsize=9)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL,
               labelcolor=MUTED_COL, loc='upper right')
    style_ax(ax1, 'Pointing error from desired attitude')

    # Annotate settling time
    settled_idx = np.argmax(error_deg < 1.0)
    if settled_idx > 0:
        ax1.axvline(t[settled_idx], color=AMBER, linewidth=0.8, linestyle=':', alpha=0.6)
        ax1.text(t[settled_idx] + 2, error_deg.max() * 0.6,
                 f'< 1° at t={t[settled_idx]:.0f}s',
                 color=AMBER, fontsize=8, alpha=0.8)

    # ── Panel 2: Angular velocity ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    for i, (col, lbl) in enumerate(zip(COLS, ['ωx', 'ωy', 'ωz'])):
        ax2.plot(t, omega_deg[:, i], color=col, linewidth=1.2, label=lbl, alpha=0.9)
    ax2.axhline(0, color=GRID_COL, linewidth=0.5)
    ax2.set_ylabel('Angular velocity (°/s)', fontsize=9)
    ax2.set_xlabel('Time (s)', fontsize=9)
    ax2.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=MUTED_COL)
    style_ax(ax2, 'Angular velocity — body frame')

    # ── Panel 3: Quaternion components ───────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    q_cols  = ['#aaaaaa', '#ff6b6b', '#51cf66', '#74c0fc']
    q_lbls  = ['w', 'x', 'y', 'z']
    for i in range(4):
        ax3.plot(t, results.q[:, i], color=q_cols[i], linewidth=1.0,
                 label=q_lbls[i], alpha=0.9)
    ax3.set_ylim(-1.1, 1.1)
    ax3.axhline(0, color=GRID_COL, linewidth=0.5)
    ax3.set_ylabel('Quaternion components', fontsize=9)
    ax3.set_xlabel('Time (s)', fontsize=9)
    ax3.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=MUTED_COL)
    style_ax(ax3, 'Attitude quaternion')

    # ── Panel 4: Control torque ───────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    for i, (col, lbl) in enumerate(zip(COLS, ['τx', 'τy', 'τz'])):
        ax4.plot(t, results.tau_ctrl[:, i] * 1000, color=col,
                 linewidth=1.0, label=lbl, alpha=0.7)
    ax4.axhline(0, color=GRID_COL, linewidth=0.5)
    ax4.set_ylabel('Control torque (mN·m)', fontsize=9)
    ax4.set_xlabel('Time (s)', fontsize=9)
    ax4.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=MUTED_COL)
    style_ax(ax4, 'Control torque — PD output')

    # ── Panel 5: Disturbance torque magnitude ─────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    dist_mag = np.linalg.norm(results.tau_dist, axis=1) * 1e6   # N·m → μN·m
    ctrl_mag = np.linalg.norm(results.tau_ctrl, axis=1) * 1000  # N·m → mN·m
    ax5.fill_between(t, dist_mag, alpha=0.2, color=ORANGE)
    ax5.plot(t, dist_mag, color=ORANGE, linewidth=1.2, label='|disturbance| (μN·m)')
    ax5_r = ax5.twinx()
    ax5_r.plot(t, ctrl_mag, color=COLS[2], linewidth=1.2,
               alpha=0.8, label='|control| (mN·m)')
    ax5_r.tick_params(colors=MUTED_COL, labelsize=9)
    ax5_r.spines['right'].set_color(GRID_COL)
    ax5_r.yaxis.label.set_color(TEXT_COL)
    for label in ax5_r.get_yticklabels():
        label.set_color(MUTED_COL)
    ax5.set_ylabel('Disturbance (μN·m)', fontsize=9)
    ax5_r.set_ylabel('Control (mN·m)', fontsize=9, color=TEXT_COL)
    ax5.set_xlabel('Time (s)', fontsize=9)
    lines1, lbls1 = ax5.get_legend_handles_labels()
    lines2, lbls2 = ax5_r.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, lbls1 + lbls2, fontsize=7,
               facecolor=PANEL_BG, edgecolor=GRID_COL, labelcolor=MUTED_COL)
    style_ax(ax5, 'Disturbance vs control torque magnitude')

    # ── Title ─────────────────────────────────────────────────
    fig.suptitle(
        '6-DOF Attitude Control Simulator  ·  PD Controller  ·  All Disturbances',
        color=TEXT_COL, fontsize=12, fontweight='normal', y=0.96
    )
    fig.patch.set_facecolor(DARK_BG)

    plt.savefig(filename, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
#  3. Reflection Geometry Plot
# ─────────────────────────────────────────────

def make_reflection_geometry(filename='reflection_geometry.png'):
    """
    Sunlight reflection mission context visualization.

    Shows:
      - Satellite elevation over time from ground target
      - Reflection angle geometry

    Connects the simulator to a real-world sunlight redirection use case.
    """
    print("Rendering reflection geometry plot...")

    ALT    = 500.0
    r      = orbital_radius(ALT)
    v      = orbital_velocity(ALT)
    T_orb  = 2 * np.pi * r / v          # orbital period (s)
    t_arr  = np.linspace(0, T_orb * 1.5, 2000)

    omega_orb = v / r
    # Ground target: Mammoth Solar Project, Starke County, Indiana
    # 41.3°N, 86.6°W
    target_lat = np.radians(41.3)
    target_lon = np.radians(-86.6)
    target_ecef = np.array([
        np.cos(target_lat) * np.cos(target_lon),
        np.cos(target_lat) * np.sin(target_lon),
        np.sin(target_lat),
    ])

    # Satellite ground track (equatorial orbit approximation)
    sat_lons = np.degrees(omega_orb * t_arr) % 360 - 180
    sat_lats = np.zeros_like(sat_lons)   # equatorial

    # Sun vector (slowly moving)
    sun_vecs = np.array([sun_vector_inertial(t) for t in t_arr])

    # Satellite position in ECI
    sat_pos = np.column_stack([
        np.cos(omega_orb * t_arr),
        np.sin(omega_orb * t_arr),
        np.zeros(len(t_arr)),
    ])

    # --- Compute access windows ---
    # Condition 1: satellite is sunlit (not in Earth's shadow)
    # Shadow: satellite is behind Earth relative to sun
    shadow_mask = np.array([
        np.dot(sat_pos[i], sun_vecs[i]) > -np.sqrt(
            max(0, 1 - (RE_EARTH / r)**2)
        )
        for i in range(len(t_arr))
    ])

    # Condition 2: satellite can geometrically reflect to target
    # Reflection condition: sun, satellite, target form valid specular geometry
    # Simplified: check if satellite elevation from target > 10°
    target_3d = target_ecef * RE_EARTH
    sat_3d    = sat_pos * r
    to_sat    = sat_3d - target_3d[np.newaxis, :]
    dist_to_sat = np.linalg.norm(to_sat, axis=1)
    to_sat_norm = to_sat / dist_to_sat[:, np.newaxis]

    # Elevation angle of satellite from target
    elevation = np.degrees(np.arcsin(
        np.clip(np.sum(to_sat_norm * target_ecef[np.newaxis, :], axis=1), -1, 1)
    ))
    elevation_mask = elevation > 10.0

    # Specular reflection check (simplified)
    # Reflection works when bisector of sun–satellite–target angle is near nadir
    specular_mask = np.array([
        np.dot(sun_vecs[i], to_sat_norm[i]) > 0.3
        for i in range(len(t_arr))
    ])

    access = shadow_mask & elevation_mask & specular_mask

    # ── Plot ─────────────────────────────────────────────────
    DARK_BG   = '#0d1117'
    PANEL_BG  = '#161b22'
    GRID_COL  = '#21262d'
    TEXT_COL  = '#c9d1d9'
    MUTED_COL = '#8b949e'

    fig = plt.figure(figsize=(10, 8), facecolor=DARK_BG)
    ax3 = fig.add_axes([0.05, 0.08, 0.90, 0.82])
    ax3.set_facecolor(PANEL_BG)
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-0.5, 2.5)
    ax3.set_aspect('equal')
    ax3.axis('off')

    # Earth
    earth = plt.Circle((0, 0), 0.55, color='#1a3a5a', zorder=1)
    ax3.add_patch(earth)
    earth_outline = plt.Circle((0, 0), 0.55, fill=False,
                                edgecolor='#3a6a9a', linewidth=1.5)
    ax3.add_patch(earth_outline)
    ax3.text(0, 0, 'Earth', color='#74c0fc', ha='center', va='center',
             fontsize=9, fontweight='bold')

    # Satellite
    sat_x, sat_y = 0.6, 1.3
    ax3.plot(sat_x, sat_y, 's', color='#74c0fc', markersize=10, zorder=5)
    ax3.text(sat_x + 0.1, sat_y + 0.1, 'Satellite', color='#74c0fc', fontsize=8)

    # Target on surface — position derived from actual coordinates
    # 41.3°N, 86.6°W: x = -R*cos(lat) (western hemisphere), y = R*sin(lat)
    R_earth_schematic = 0.55
    tgt_x = -R_earth_schematic * np.cos(np.radians(41.3))   # -0.414
    tgt_y =  R_earth_schematic * np.sin(np.radians(41.3))   #  0.363
    ax3.plot(tgt_x, tgt_y, 'o', color='#ff6b6b', markersize=8, zorder=5)
    ax3.text(tgt_x - 0.15, tgt_y + 0.12, 'Mammoth\nSolar\nProject',
             color='#ff6b6b', fontsize=7)

    # Sun
    sun_x, sun_y = -1.7, 1.8
    ax3.plot(sun_x, sun_y, '*', color='#ffd43b', markersize=16, zorder=5)
    ax3.text(sun_x + 0.1, sun_y, 'Sun', color='#ffd43b', fontsize=9)

    # Incoming sunlight
    ax3.annotate('', xy=(sat_x, sat_y),
                 xytext=(sun_x + 0.25, sun_y - 0.18),
                 arrowprops=dict(arrowstyle='->', color='#ffd43b',
                                 lw=1.5, alpha=0.8))
    ax3.text(-0.7, 1.65, 'incident\nlight', color='#ffd43b',
             fontsize=7, ha='center', alpha=0.8)

    # Reflected beam
    ax3.annotate('', xy=(tgt_x, tgt_y + 0.05),
                 xytext=(sat_x - 0.04, sat_y - 0.1),
                 arrowprops=dict(arrowstyle='->', color='#ff922b',
                                 lw=2.0, alpha=0.9))
    ax3.text(0.25, 0.95, 'reflected\nbeam', color='#ff922b',
             fontsize=7, ha='center', alpha=0.9)

    # Mirror on satellite
    mirror_dx, mirror_dy = 0.18, 0.06
    ax3.plot([sat_x - mirror_dx, sat_x + mirror_dx],
             [sat_y - mirror_dy, sat_y + mirror_dy],
             color='#c0e0ff', linewidth=3, alpha=0.9, zorder=6)

    # Pointing accuracy annotation
    ax3.annotate('Pointing error\nmust be < 0.1°\nfor 1km spot size',
                 xy=(sat_x, sat_y), xytext=(1.1, 0.8),
                 color=MUTED_COL, fontsize=7,
                 arrowprops=dict(arrowstyle='->', color=MUTED_COL,
                                 lw=0.8, alpha=0.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL_BG,
                           edgecolor=GRID_COL, alpha=0.8))

    ax3.set_title('Reflection geometry — why pointing accuracy matters',
                  color=TEXT_COL, fontsize=10, pad=8)

    # ── Title ─────────────────────────────────────────────────
    fig.suptitle(
        'LEO Sunlight Reflection Geometry · Target: Mammoth Solar Project, Indiana',
        color=TEXT_COL, fontsize=12, y=0.96
    )
    fig.patch.set_facecolor(DARK_BG)

    plt.savefig(filename, dpi=150, facecolor=DARK_BG, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Run simulation once — shared by plots 1 and 2
    results, cfg, ctrl = run_showcase_simulation()

    # 1. Animated GIF
    make_animation(results, ctrl, filename='animated_attitude.gif', n_frames=160)

    # 2. Summary plot
    make_summary_plot(results, ctrl, filename='summary_plot.png')

    # 3. Reflection geometry (standalone — no simulation needed)
    make_reflection_geometry(filename='reflection_geometry.png')

    print()
    print("All outputs saved:")
    print("  animated_attitude.gif   ← put this at the top of your README")
    print("  summary_plot.png        ← technical results panel")
    print("  reflection_geometry.png ← sunlight reflection mission context")
    print()
    print("GitHub README tip:")
    print("  ![Attitude Control Demo](animated_attitude.gif)")