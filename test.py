import numpy as np
import matplotlib.pyplot as plt
from config import Ising2DTF

res = 20
Bs = np.linspace(0, 100, res)
Is = np.linspace(0, 100, res)
M = np.zeros((res,res))


T = 100.
for n,B in enumerate(Bs):
    for m,I in enumerate(Is):
        model = Ising2DTF(5, B=B, I=I, T=T)
        model.evolve(100)
        M[n][m] = np.abs(model.M())

plt.imshow(M)
plt.colorbar()
plt.xlabel("Transverse field (B)")
plt.ylabel("Interaction strength (I)")

plt.show()


