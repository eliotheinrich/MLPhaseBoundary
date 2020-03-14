import numpy as np
import matplotlib.pyplot as plt
from config import Ising2DTF

res = 20
Bs = np.linspace(0.1, 1000, res)
Is = np.linspace(0.1, 1000, res)
Ts = np.linspace(0.1, 1000, res)
L = 20
N = 200
m = 10000
X = np.zeros((m, 4))

for i in range(m):
    X[i][0] = 100 - 100*np.random.rand() # B
    X[i][1] = 100 - 100*np.random.rand() # I
    X[i][2] = 100 - 100*np.random.rand() # T
    
    model = Ising2DTF(L, B=X[i][0], I=X[i][1], T=X[i][2])
    model.evolve(N)
    
    X[i][3] = np.abs(model.M())
    
    if (i%100 == 1):
        print(i)
        
        
np.savetxt("data/Magnetization_data1.txt", X, header =
           '%22s %24s %24s %24s' % ('B', 'I', 'T', 'M'))