import numpy as np
import matplotlib.pyplot as plt

# choose scenario: "descent", "circular", "elliptic", "escape"
scenario = "circular"

# use symplectic euler for better orbital stability (True) or standard euler (False)
use_symplectic = True

# constants from the handout (Mars)
G = 6.674e-11           # gravitational constant
M = 6.42e23             # mass of Mars (kg)
r0 = 3.386e6            # Mars radius (m)
mu = G * M

# test mass and initial radius/height
m = 1000.0
h = 0.0
r_start = r0 + h

# initial position vector (start on +x axis)
s = np.array([r_start, 0.0, 0.0])

# pick initial velocity based on scenario
if scenario == "descent":
    v = np.array([0.0, 0.0, 0.0])
else:
    v_circ = np.sqrt(mu / r_start)  # circular speed at r_start
    if scenario == "circular":
        v_mag = v_circ
    elif scenario == "elliptic":
        v_mag = 0.8 * v_circ
    elif scenario == "escape":
        v_mag = np.sqrt(2.0) * v_circ
    else:
        raise ValueError("scenario must be one of: descent, circular, elliptic, escape")
    # velocity perpendicular to radius (along +y)
    v = np.array([0.0, v_mag, 0.0])

# simulation time, timestep and time array
t_max = 50000
dt = 1.0
t_array = np.arange(0, t_max, dt)

# storage
s_list = []
alt_list = []  # for descent case

# integration loop
for t in t_array:
    # store current state
    s_list.append(s.copy())
    if scenario == "descent":
        alt_list.append(np.linalg.norm(s) - r0)

    # gravitational acceleration a = -mu * r / |r|^3
    r_vec = s
    r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]
    r_mag = np.sqrt(r2)
    a = -mu * r_vec / (r_mag**3)

    # update (symplectic euler is more stable for orbits)
    if use_symplectic:
        v = v + dt * a
        s = s + dt * v
    else:
        s = s + dt * v
        v = v + dt * a

# arrays for plotting
s_x_array = np.array([p[0] for p in s_list])
s_y_array = np.array([p[1] for p in s_list])

# plot
if scenario == "descent":
    plt.title('DESCENT: altitude vs time (dt = {}, method = {})'.format(
        dt, 'symplectic' if use_symplectic else 'euler'))
    plt.xlabel('time (s)')
    plt.ylabel('altitude (m)')
    plt.grid()
    plt.plot(t_array, np.array(alt_list), label='altitude')
    plt.legend()
else:
    plt.title('ORBIT: {} (dt = {}, method = {})'.format(
        scenario, dt, 'symplectic' if use_symplectic else 'euler'))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.grid()
    plt.plot(s_x_array, s_y_array, label='trajectory')
    plt.scatter([0.0], [0.0], s=30, label='planet')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()

plt.show()
