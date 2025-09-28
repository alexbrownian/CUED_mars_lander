# HOW TO USE:
#   1) Pick MODE = "A1" or "A2"
#   2) For A1: choose TASK_A1 = 1, 2, or 3
#   3) For A2: choose SCENARIO = "descent" / "circular" / "elliptic" / "escape"

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# =============================================
# CHOOSE WHAT TO RUN
# =============================================
MODE = "A1"            # "A1" for Assignment 1, "A2" for Assignment 2
TASK_A1 = 2            # 1: basic Euler spring; 2: compare Euler vs Verlet and timesteps; 3: find critical dt for Verlet
SCENARIO = "circular"  # A2 only: "descent", "circular", "elliptic", "escape"

# =============================================
# STEP A1.0 — COMMON SPRING SETUP (used by all A1 tasks)
# =============================================
m = 1.0          # mass
k = 1.0          # spring constant
x0 = 0.0         # initial position
v0 = 1.0         # initial velocity
t_max = 1000.0   # total time to simulate
dt_default = 0.01

# handy analytic freq/period for comparisons in A1
omega = sqrt(k/m)
period = 2.0*np.pi/omega

# =============================================
# STEP A1.1 — EULER INTEGRATION FOR 1D SPRING (Task 1)
# =============================================
def euler_spring(x0, v0, m, k, dt, t_max):
    xs, vs, ts = [], [], []
    x, v, t = x0, v0, 0.0
    while t < t_max:
        xs.append(x); vs.append(v); ts.append(t)
        # acceleration = F/m = -kx/m
        a = -k*x/m
        # Euler step
        x = x + dt*v
        v = v + dt*a
        t += dt
    return np.array(ts), np.array(xs), np.array(vs)

# =============================================
# STEP A1.2 — VERLET INTEGRATION FOR 1D SPRING (Task 1 + 2)
# =============================================
def verlet_spring(x0, v0, m, k, dt, t_max):
    # Need x at t=0 and t=-dt. Use a backward Euler half-step for bootstrap.
    xs, vs, ts = [], [], []
    t = 0.0
    x_curr = x0
    a_curr = -k*x_curr/m
    # estimate previous position using x_{-dt} = x0 - v0*dt + 0.5*a0*dt^2
    x_prev = x_curr - v0*dt + 0.5*a_curr*(dt*dt)

    while t < t_max:
        xs.append(x_curr); ts.append(t)
        # Verlet position update: x_next = 2*x_curr - x_prev + a_curr*dt^2
        x_next = 2.0*x_curr - x_prev + a_curr*(dt*dt)
        # estimate velocity (central difference, 1 step behind)
        v_est = (x_next - x_prev)/(2.0*dt)
        vs.append(v_est)
        # roll forward
        x_prev, x_curr = x_curr, x_next
        a_curr = -k*x_curr/m
        t += dt

    return np.array(ts), np.array(xs), np.array(vs)

# =============================================
# STEP A1.3 — ANALYTICAL SOLUTION (for checking Task 1/2)
# =============================================
def analytic_solution(ts, x0, v0, omega):
    # x(t) = A cos(omega t) + B sin(omega t); with x(0)=x0, v(0)=v0
    A = x0
    B = v0/omega
    xs = A*np.cos(omega*ts) + B*np.sin(omega*ts)
    vs = -A*omega*np.sin(omega*ts) + B*omega*np.cos(omega*ts)
    return xs, vs

# =============================================
# STEP A1.4 — TIMESTEP SWEEP + STABILITY CHECKS (Task 2/3)
# =============================================
def run_timestep_sweep(ts_method="verlet", dt_list=(0.5,1.0,2.0), t_max=1000.0):
    results = {}
    for dt in dt_list:
        if ts_method == "verlet":
            ts, xs, vs = verlet_spring(x0, v0, m, k, dt, t_max)
        else:
            ts, xs, vs = euler_spring(x0, v0, m, k, dt, t_max)
        results[dt] = (ts, xs, vs)
    return results

def find_critical_dt_verlet(start=0.1, stop=3.0, step=0.1, thresh=10.0, t_max=1000.0):
    # very rough "instability" test: if |x| ever exceeds thresh * max(|x analytic|) we call it unstable
    # for m=k=v0=1, analytic amplitude is ~sqrt(A^2 + B^2) ~ sqrt(x0^2 + (v0/omega)^2)
    Aamp = sqrt(x0*x0 + (v0/omega)**2)
    for dt in np.arange(start, stop+1e-9, step):
        ts, xs, vs = verlet_spring(x0, v0, m, k, dt, t_max)
        if np.max(np.abs(xs)) > thresh*Aamp:
            return dt
    return None

# =============================================
# STEP A2.0 — COMMON ORBIT SETUP (used by all A2 scenarios)
# =============================================
G = 6.674e-11
M = 6.42e23        # Mars mass - CONSTANT
mu = G*M
r_mars = 3.39e6

def gravity_accel(r_vec):
    r2 = np.dot(r_vec, r_vec)
    r = np.sqrt(r2)
    if r == 0.0:
        return np.array([0.0,0.0,0.0])
    return -mu * r_vec / (r*r*r)

# simple Euler and simple symplectic Euler (velocity first) in 3D
def euler_orbit_step(s, v, dt):
    a = gravity_accel(s)
    s_new = s + dt*v
    v_new = v + dt*a
    return s_new, v_new

def symplectic_orbit_step(s, v, dt):
    a = gravity_accel(s)
    v_new = v + dt*a
    s_new = s + dt*v_new
    return s_new, v_new

# =============================================
# STEP A2.1 — RUN SCENARIOS (Task list 1–4)
# =============================================
def run_orbit(scenario="circular", dt=1.0, t_max=50000.0, method="symplectic"):
    s = np.array([r_mars, 0.0, 0.0], dtype=float)  # start on x-axis at planet radius
    # choose initial v perpendicular to s for 2–4; zero for descent
    if scenario == "descent":
        v = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        v_circ = sqrt(mu/np.linalg.norm(s))
        if scenario == "circular":
            v_mag = v_circ
        elif scenario == "elliptic":
            v_mag = 0.8*v_circ   # less than circular -> ellipse
        elif scenario == "escape":
            v_mag = sqrt(2)*v_circ  # escape speed at r0
        else:
            raise ValueError("unknown scenario")
        v = np.array([0.0, v_mag, 0.0], dtype=float)

    stepper = symplectic_orbit_step if method=="symplectic" else euler_orbit_step
    xs, ys, alts, ts = [], [], [], []
    t = 0.0
    while t < t_max:
        xs.append(s[0]); ys.append(s[1])
        alts.append(np.linalg.norm(s) - r_mars)
        ts.append(t)
        s, v = stepper(s, v, dt)
        t += dt
    return np.array(ts), np.array(xs), np.array(ys), np.array(alts)

# =============================================
# RUNNER — Makes the plots for each task
# =============================================
if MODE == "A1":
    if TASK_A1 == 1:
        # Task 1: implement Euler and Verlet, and plot x(t) and compare to analytic
        dt = 0.01
        ts_e, xs_e, vs_e = euler_spring(x0, v0, m, k, dt, 50.0)
        ts_v, xs_v, vs_v = verlet_spring(x0, v0, m, k, dt, 50.0)
        xs_an, vs_an = analytic_solution(ts_v, x0, v0, omega)

        plt.figure()
        plt.title("A1 Task 1: Spring x(t) — Euler vs Verlet vs Analytic")
        plt.plot(ts_e, xs_e, label="Euler x(t)")
        plt.plot(ts_v, xs_v, label="Verlet x(t)")
        plt.plot(ts_v, xs_an, label="Analytic x(t)", linestyle="--")
        plt.xlabel("t")
        plt.ylabel("x")
        plt.grid(True)
        plt.legend()
        plt.show()

    elif TASK_A1 == 2:
        # Task 2: investigate different dt and compare algorithms
        dt_list = [0.5, 1.0, 2.0]
        results_verlet = run_timestep_sweep("verlet", dt_list=dt_list, t_max=1000.0)
        results_euler  = run_timestep_sweep("euler",  dt_list=dt_list, t_max=100.0)  # shorter for Euler drift

        # Plot a small time window for visibility
        plt.figure()
        plt.title("A1 Task 2: Verlet — effect of dt on x(t)")
        for dt, (ts, xs, vs) in results_verlet.items():
            plt.plot(ts, xs, label=f"Verlet dt={dt}")
        plt.xlim(0, 200)
        plt.xlabel("t"); plt.ylabel("x")
        plt.grid(True); plt.legend()
        plt.show()

        plt.figure()
        plt.title("A1 Task 2: Euler — effect of dt on x(t)")
        for dt, (ts, xs, vs) in results_euler.items():
            plt.plot(ts, xs, label=f"Euler dt={dt}")
        plt.xlim(0, 50)
        plt.xlabel("t"); plt.ylabel("x")
        plt.grid(True); plt.legend()
        plt.show()

    elif TASK_A1 == 3:
        # Task 3: find (rough) critical dt for Verlet instability
        crit = find_critical_dt_verlet(start=0.5, stop=3.0, step=0.1, t_max=1000.0)
        print("Approx critical dt for Verlet instability:", crit)

        # Quick illustration at "stable" and "unstable"
        stable_dt = 1.0
        unstable_dt = 2.0
        ts_s, xs_s, _ = verlet_spring(x0, v0, m, k, stable_dt, 500.0)
        ts_u, xs_u, _ = verlet_spring(x0, v0, m, k, unstable_dt, 500.0)

        plt.figure()
        plt.title("A1 Task 3: Verlet stability demo")
        plt.plot(ts_s, xs_s, label=f"stable dt={stable_dt}")
        plt.plot(ts_u, xs_u, label=f"unstable dt={unstable_dt}")
        plt.xlabel("t"); plt.ylabel("x")
        plt.grid(True); plt.legend()
        plt.show()

elif MODE == "A2":
    # Assignment 2: scenarios 1–4
    if SCENARIO == "descent":
        ts, xs, ys, alts = run_orbit("descent", dt=1.0, t_max=20000.0, method="symplectic")
        plt.figure()
        plt.title("A2 Scenario 1: Straight-down Descent — Altitude vs Time")
        plt.plot(ts, alts, label="altitude")
        plt.xlabel("t [s]"); plt.ylabel("altitude [m]")
        plt.grid(True); plt.legend()
        plt.show()
    else:
        # scenarios 2–4: plot trajectory in orbital plane
        ts, xs, ys, alts = run_orbit(SCENARIO, dt=1.0, t_max=50000.0, method="symplectic")
        plt.figure()
        plt.title(f"A2 Scenario: {SCENARIO} — trajectory in orbital plane")
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.grid(True)
        plt.plot(xs, ys, label="trajectory")
        plt.scatter([0.0],[0.0], s=30, label="planet")
        plt.xlabel("x [m]"); plt.ylabel("y [m]")
        plt.legend()
        plt.show()

# code_text = open('/mnt/data/assignments_combined.py','w', encoding='utf-8')
# code_text.write(open(__file__, 'r', encoding='utf-8').read())
# code_text.close()
# print("Saved script to /mnt/data/assignments_combined.py")
