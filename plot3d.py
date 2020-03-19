import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import Ising2DTF


T_min = 98
T_max = 99
data = np.loadtxt("data/Magnetization_data2.txt")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in data:
    if i[3] > .5:
        ax.scatter(i[0], i[1], i[2], color='r')
    else:
        ax.scatter(i[0], i[1], i[2], color='b')


ax.set_xlabel("Magnetic field (B)")
ax.set_ylabel("Interaction strength (I)")
ax.set_zlabel("Temperature (T)")
plt.show()


