import numpy as np
import matplotlib.pyplot as plt

# mass, spring constant, initial position and velocity
m = 1
k = 1
x = 0 #displacement
v = 1

# simulation time, timestep and time
t_max = 100
dt = 2 #COMMENT: it actually is stable for 1 but not for 2
t_array = np.arange(0, t_max, dt)

# initialise empty lists to record trajectories
x_list = []
v_list = []

# Verlet integration
for i in range(len(t_array)):

    # append current state to trajectories
    x_list.append(x)
    v_list.append(v)

    if i == 0:
        x_prev = x - v*dt + 0.5 * (-k * x / m) * dt * dt  # general bootstrap (central diff theorem)
    else:
        x_prev = x_list[i-1]

    a = -k * x / m
    x_current = x
    x = 2 * x - x_prev + dt * dt * a
    v = (x - x_prev) / (2*dt)  # central difference velocity

# convert trajectory lists into arrays, so they can be sliced (useful for Assignment 2)
x_array = np.array(x_list)
v_array = np.array(v_list)

# plot the position-time graph
plt.title('VERLET, dt = {}'.format(dt))
plt.xlabel('time (s)')
plt.grid()
plt.plot(t_array, x_array, label='x (m)')
plt.plot(t_array, v_array, label='v (m/s)')
plt.legend()
plt.show()