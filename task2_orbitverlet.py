import numpy as np
import matplotlib.pyplot as plt

# mass, spring constant, initial position and velocity
G = 6.674e-11         # from PDF
M = 6.42e23           # from PDF (Mars)
m = 1000
GMm = G * M * m

h = 0
r0 = 3.386e6          # from PDF (Mars radius)
e = 1

s = np.array([r0 + h, 0, 0])
v = np.array([0, e*np.sqrt(G*M/(r0+h)), 0])

# simulation time, timestep and time
t_max = 50000
dt = 1
t_array = np.arange(0, t_max, dt)

# initialise empty lists to record trajectories
s_list = []

for i in range(len(t_array)):

    # append current state to trajectories
    s_list.append(s.copy())

    # calculate new position and velocity
    r = s - np.array([0, 0, 0])
    r_mag = np.linalg.norm(r)
    r_norm = r / r_mag

    F_mag = -GMm / (r_mag ** 2)
    F = r_norm * F_mag

    if i == 0:
        a0 = F / m
        s_prev = s - v * dt + 0.5 * a0 * (dt * dt)  # general bootstrap at i=0
    else:
        s_prev = s_list[i-1]

    a = F / m
    s_current = s
    s = 2 * s - s_prev + dt * dt * a
    v = (s - s_prev) / (2 * dt)  # central-difference velocity for Verlet

# convert trajectory lists into arrays, so they can be sliced (useful for Assignment 2)
s_x_array = np.array([s_list[i][0] for i in range(len(s_list))])
s_y_array = np.array([s_list[i][1] for i in range(len(s_list))])
s_z_array = np.array([s_list[i][2] for i in range(len(s_list))])

# plot the position-time graph
plt.title('VERLET ORBIT, dt = {}, e = {}'.format(dt, e))
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.plot(s_x_array, s_y_array, label='position')
plt.legend()
plt.show()
