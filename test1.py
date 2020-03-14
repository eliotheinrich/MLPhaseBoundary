import numpy as np
import matplotlib.pyplot as plt
from config import Ising2DTF

res = 20
Bs = np.linspace(0.1, 1000, res)
Is = np.linspace(0.1, 1000, res)
Ts = np.linspace(0.1, 1000, res)
M = np.zeros((res,res))
L = 20

B = 10.
for n,T in enumerate(Ts):
    for m,I in enumerate(Is):
        model = Ising2DTF(L, B=B, I=I, T=T)
        model.evolve(100)
        M[n][m] = np.abs(model.M())
    print(n)

np.savetxt("data/I_T_Magnetization.txt", M)

plt.imshow(M.transpose(), origin = "lower")
plt.colorbar()
plt.title("B = 10")
plt.xlabel("Temperature (T)")
plt.ylabel("Interaction strength (I)")
plt.savefig("plots/I_T_phase_diagram.png")
plt.show()

I = 1.
for n,B in enumerate(Bs):
    for m,T in enumerate(Ts):
        model = Ising2DTF(L, B=B, I=I, T=T)
        model.evolve(100)
        M[n][m] = np.abs(model.M())
    print(n)
    
np.savetxt("data/B_T_Magnetization.txt", M)

plt.imshow(M.transpose(), origin = "lower")
plt.colorbar()
plt.title("I = 1")
plt.xlabel("Transverse field (B)")
plt.ylabel("Temperature (T)")
plt.savefig("plots/B_T_phase_diagram.png")
plt.show()

T = 100.
for n,B in enumerate(Bs):
    for m,I in enumerate(Is):
        model = Ising2DTF(L, B=B, I=I, T=T)
        model.evolve(100)
        M[n][m] = np.abs(model.M())
    print(n)
    
np.savetxt("data/I_B_Magnetization.txt", M)

plt.imshow(M.transpose(), origin = "lower")
plt.colorbar()
plt.title("T = 100")
plt.xlabel("Transverse field (B)")
plt.ylabel("Interaction strength (I)")
plt.savefig("plots/I_B_phase_diagram.png")
plt.show()