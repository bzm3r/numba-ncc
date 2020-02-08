import numpy as np
import matplotlib.pyplot as plt
import copy

def generate_particle_data(num_trials, num_tpoints, persistence, std_persistence, init_v, init_p, avg_dv, std_dv):
    ps = np.zeros((num_trials, num_tpoints, 2), dtype=np.float64)
    ps[:, 0, :] = init_p*np.ones((num_trials, 2))



num_trials = 1
num_tpoints = 20000

init_p_a = np.zeros(2, dtype=np.float64)
init_v_a = np.array([0.0, 0.0])
std_dv_a = np.array([1.0, 0.1])
dvs, vs, ps = generate_particle_data(num_trials, num_tpoints, init_v_a, init_p_a, std_dv_a)

fig, ax = plt.subplots()

for nix in range(num_trials):
    ax.plot(ps[nix, :, 0], ps[nix, :, 1], marker=".")

ax.set_aspect("equal")
plt.show()
