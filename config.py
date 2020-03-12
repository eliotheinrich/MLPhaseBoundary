import numpy as np

class Ising2DTF:
    def __init__(self, L, B, I, T):
        self.L = L
        self.B = B
        self.I = I
        self.T = T
        self.s = np.random.choice([-1,1], size=(L,L))

    # Computes magnetization per spin of system
    def M(self):
        return np.sum(self.s)/self.L**2

    # Computes internal energy of system
    def E(self):
        E = 0.
        for i in range(0, self.L):
            for j in range(0, self.L):
                # Transverse field contribution
                E += self.B*self.s[i][j]
                # Interaction contribution
                E -= self.I/2.*self.s[i][j]*(self.s[i][(j+1)%self.L]
                                   + self.s[i][(j-1)%self.L]
                                   + self.s[(i+1)%self.L][j]
                                   + self.s[(i-1)%self.L][j])

        return E

    # Computes change in internal energy from flipping spin at site (i,j)
    def dE(self, i, j):
        dE = 0.
        dE -= 2*self.B*self.s[i][j]
        dE += 2*self.I*self.s[i][j]*(self.s[i][(j+1)%self.L]
                                   + self.s[i][(j-1)%self.L]
                                   + self.s[(i+1)%self.L][j]
                                   + self.s[(i-1)%self.L][j])
        return dE

    # Flips spin at site (i,j)
    def flip(self, i, j):
        self.s[i][j] = -self.s[i][j]


    # Does N steps of Monte-Carlo evolution
    def evolve(self, N=1):
        for n in range(0, N):
            for m in range(0, self.L**2): 
                i = np.random.randint(0, self.L)
                j = np.random.randint(0, self.L)
                r = np.random.random()
                delta = self.dE(i,j)
                P = np.exp(-delta/self.T)
                # If move is accepted, flip spin at site (i,j)
                if r < P:
                    self.flip(i,j)

    def __getitem__(self,key):
        return self.s[key]

    def __str__(self):
        return str(self.s)


