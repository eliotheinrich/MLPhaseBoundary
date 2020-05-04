import numpy as np
import matplotlib.pyplot as plt
import pickle
from config import Ising2DTF
from config import TriangularIsing2DTF
from config import HexagonalIsing2DTF
res = 10
Bs = np.linspace(0.1, 1000, res)
Is = np.linspace(0.1, 1000, res)
Ts = np.linspace(0.1, 1000, res)
L = 20
N = 1000
m = 10000
X = np.zeros((m, 4))
Final_States = np.zeros((m, L, L))

for i in range(m):
    X[i][0] = 60 - 60*np.random.rand() # B
    X[i][1] = 100 - 100*np.random.rand() # I
    X[i][2] = 100 - 100*np.random.rand() # T
    
    model = HexagonalIsing2DTF(L, B=X[i][0], I=X[i][1], T=X[i][2])
    model.evolve(N)
    
    X[i][3] = np.abs(model.M())
    
    Final_States[i] = model.s
    
    if (i%100 == 1):
        print(i)
        
state_file = open(r'Ising_state_hex2.pkl', 'wb')
pickle.dump(Final_States, state_file)
state_file.close()
        
np.savetxt("data/Magnetization_data_hex2.txt", X, header =
           '%22s %24s %24s %24s' % ('B', 'I', 'T', 'M'))