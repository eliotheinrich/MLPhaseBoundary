import pickle
import matplotlib.pyplot as plt
from plot_lattice import plot_checkerboard

with open(r'Ising_state_gauge.pkl', 'rb') as f: 
    gauge_states = pickle.load(f)

with open(r'Ising_state_ice.pkl', 'rb') as f: 
    ice_states = pickle.load(f)

with open(r'gauge_hist.pkl', 'rb') as f: 
    gauge_hist = pickle.load(f)

with open(r'ice_hist.pkl', 'rb') as f: 
    ice_hist = pickle.load(f)


fig,ax = plt.subplots(nrows=1, ncols=2)
plt.subplots_adjust(wspace=0)

ax[0].plot(ice_hist['val_binary_accuracy'], 'k-', label="Validation accuracy")
ax[0].plot(ice_hist['binary_accuracy'], 'r--', label="Training accuracy")
ax[0].set_xlabel("Epochs")
ax[0].set_title("Square ice training accuracy")
ax[0].set_xlim(0, len(ice_hist['binary_accuracy'])-1)
plt.grid()

ax[1].plot(gauge_hist['val_binary_accuracy'], 'k-', label="Validation accuracy")
ax[1].plot(gauge_hist['binary_accuracy'], 'r--', label="Training accuracy")
ax[1].legend(fontsize=15)
ax[1].set_xlabel("Epochs")
plt.grid()
ax[1].set_yticks([])
ax[1].set_title("Gauge theory training accuracy")
ax[1].set_xlim(0, len(gauge_hist['binary_accuracy'])-1)
plt.show()



fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(12,6))
plt.subplots_adjust(wspace=.05)
plt.tight_layout()

ax[0][0].set_title("Low temperature", fontsize=15)
ax[0][0].set_ylabel("Square Ice", fontsize=15)
ax[0][0].set_xticks([])
ax[0][0].set_yticks([])

ax[0][1].set_title("High temperature", fontsize=15)
ax[0][1].set_xticks([])
ax[0][1].set_yticks([])

ax[1][0].set_ylabel("Ising Gauge Theory", fontsize=15)
ax[1][0].set_xticks([])
ax[1][0].set_yticks([])

ax[1][1].set_xticks([])
ax[1][1].set_yticks([])

plot_checkerboard(ax[0][0], ice_states[10])
plot_checkerboard(ax[0][1], ice_states[75])
plot_checkerboard(ax[1][0], ice_states[10])
plot_checkerboard(ax[1][1], gauge_states[700])
plt.show()




