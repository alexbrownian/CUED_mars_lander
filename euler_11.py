import numpy as np
import matplotlib.pyplot as plt

# mass, spring constant, initial position and velocity
G = 6.674 * 10e-11
M = 6.42 * 10e23
m = 1000
GMm = G * M * m

h = 0
r0 = 6*10e6
e = 1

s = np.array([r0 + h, 0, 0])
v = np.array([0, e*np.sqrt(G*M/(r0+h)), 0])

# simulation time, timestep and time
t_max = 50000
dt = 1
t_array = np.arange(0, t_max, dt)

# initialise empty lists to record trajectories
s_list = []
# v_list = []

# Euler integration
for t in t_array:

    s_list.append(s)
    # v_list.append(v)

    # calculate new position and velocity
    r = s - np.array([0, 0, 0])
    # r_mag = np.linalg.norm(r)
    r_mag2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2]
    r_mag = np.sqrt(r_mag2)
    r_norm = r / r_mag


    F_mag = -GMm/ r_mag2
    F = r_norm * F_mag

    # append current state to trajectories
    a = F / m
    s = s + dt * v
    v = v + dt * a

# convert trajectory lists into arrays, so they can be sliced (useful for Assignment 2)
s_x_array = np.array([s_list[i][0] for i in range(len(s_list))])
s_y_array = np.array([s_list[i][1] for i in range(len(s_list))])
s_z_array = np.array([s_list[i][2] for i in range(len(s_list))])

# v_x_array = np.array([v_list[i][0] for i in range(len(v_list))])
# v_y_array = np.array([v_list[i][1] for i in range(len(v_list))])
# v_z_array = np.array([v_list[i][2] for i in range(len(v_list))])

# plot the position-time graph
plt.title('EULER ORBIT, dt = {}, e = {}'.format(dt, e))
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.plot(s_x_array, s_y_array, label='position')

plt.legend()
plt.show()
