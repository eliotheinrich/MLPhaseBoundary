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


class TriangularIsing2DTF:
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
                                           + self.s[(i-1)%self.L][j]
                                           + self.s[(i+1)%self.L][(j+1)%self.L]
                                           + self.s[(i-1)%self.L][(j+1)%self.L])

        return E

    # Computes change in internal energy from flipping spin at site (i,j)
    def dE(self, i, j):
        dE = 0.
        dE -= 2*self.B*self.s[i][j]
        dE += 2*self.I*self.s[i][j]*(self.s[i][(j+1)%self.L]
                                   + self.s[i][(j-1)%self.L]
                                   + self.s[(i+1)%self.L][j]
                                   + self.s[(i-1)%self.L][j]
                                   + self.s[(i+1)%self.L][(j+1)%self.L]
                                   + self.s[(i-1)%self.L][(j+1)%self.L])
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


class HexagonalIsing2DTF:
    def __init__(self, L, B, I, T):
        if L%4 != 0:
            L = 4*round(L/4)
            print("L must be a multiple of 4 in a hexagonal lattice. Rounding up to " + str(L))
        self.L = L
        self.B = B
        self.I = I
        self.T = T
        self.s = np.random.choice([-1,1], size=(L,L))

    # Computes magnetization per spin of system
    def M(self):
        return np.sum(self.s)/self.L**2


    # Horrible hack to get the sum of spins of the neighbors of site (i,j) on a hexagonal lattice
    def get_neighbors(self, i, j):
        S = 0.
        if i%4==0:
            if j%4==0: 
                S += self.s[i][j+1] \
                   + self.s[(i-1)%self.L][j] \
                   + self.s[i+1][j]
            elif j%4==1: 
                S += self.s[i][j-1] \
                   + self.s[i][j+1] \
                   + self.s[i+1][j]
            elif j%4==2:
                S += self.s[(i-1)%self.L][j] \
                   + self.s[i][j-1] \
                   + self.s[i][j+1]
            elif j%4==3:
                S += self.s[(i-1)%self.L][j] \
                   + self.s[i+1][j] \
                   + self.s[i][j-1]
        elif i%4==1:
            if j%4==0:
                S += self.s[i-1][j] \
                   + self.s[i+1][j] \
                   + self.s[i][(j-1)%self.L]
            elif j%4==1:
                S += self.s[i+1][j] \
                   + self.s[i-1][j] \
                   + self.s[i][j+1]
            elif j%4==2:
                S += self.s[i+1][j] \
                   + self.s[i][j-1] \
                   + self.s[i][j+1]
            elif j%4==3:
                S += self.s[i-1][j] \
                   + self.s[i][j-1] \
                   + self.s[i][(j+1)%self.L]
        elif i%4==2:
            if j%4==0:
                S += self.s[i][(j-1)%self.L] \
                   + self.s[i][j+1] \
                   + self.s[i-1][j]
            elif j%4==1:
                S += self.s[i][j-1] \
                   + self.s[i-1][j] \
                   + self.s[i+1][j]
            elif j%4==2:
                S += self.s[i-1][j] \
                   + self.s[i+1][j] \
                   + self.s[i][j+1]
            elif j%4==3:
                S += self.s[i+1][j] \
                   + self.s[i][j-1] \
                   + self.s[i][(j+1)%self.L]
        elif i%4==3:
            if j%4==0:
                S += self.s[(i+1)%self.L][j] \
                   + self.s[i][(j-1)%self.L] \
                   + self.s[i][j+1]
            elif j%4==1:
                S += self.s[i][j-1] \
                   + self.s[i][j+1] \
                   + self.s[i-1][j]
            elif j%4==2:
                S += self.s[i][j-1] \
                   + self.s[(i+1)%self.L][j] \
                   + self.s[i-1][j]
            elif j%4==3:
                S += self.s[(i+1)%self.L][j] \
                   + self.s[i-1][j] \
                   + self.s[i][(j+1)%self.L]

        return S

    # Computes internal energy of system
    def E(self):
        E = 0.
        for i in range(0, self.L):
            for j in range(0, self.L):
                # Transverse field contribution
                E += self.B*self.s[i][j]
                # Interaction contribution
                E -= self.I/2.*self.s[i][j]*self.get_neighbors(i,j)

        return E

    # Computes change in internal energy from flipping spin at site (i,j)
    def dE(self, i, j):
        dE = 0.
        dE -= 2*self.B*self.s[i][j]
        dE += 2*self.I*self.s[i][j]*self.get_neighbors(i,j)
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



model = HexagonalIsing2DTF(20, 1., 1., 1.)
model.evolve(100)





