import numpy as np
from numba import jit

class Material:

    def __init__(self, gamma, E, f_druck, f_zug):

        self.gamma = gamma
        self.E     = E
        self.n     = 0.5

        self.f_druck = f_druck
        self.f_zug   = f_zug

class Concrete_C30_37(Material):

    def __init__(self):
        gamma   = 25 * 10**(-6)  # N/mm3
        E       = 32000          # N/mm2
        f_druck = 20             # N/mm2
        f_zug   = 1.28           # N/mm2

        super().__init__(gamma, E, f_druck, f_zug)
        self.color = (0, 0, 0, 0.5)
        self.name  = "Concrete_C30_37"
        
    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        E       = 32000          # N/mm2
        f_druck = 20             # N/mm2
        f_zug   = 1.28           # N/mm2

        stresses = E * strains
        stresses = np.clip(stresses, -f_zug, f_druck)  # Efficient thresholding
        stresses = np.asarray(stresses, dtype=np.float64)
        stresses[stresses <= -f_zug] = 0
        return stresses


class Steel_S235(Material):

    def __init__(self):

        gamma   = 78.5 * 10**(-6) # N/mm3
        E       = 210000          # N/mm2
        f_druck = 235             # N/mm2
        f_zug   = 235             # N/mm2

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (0, 0, 1, 0.5)
        self.name  = "Steel_S235" 

    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        E       = 210000          # N/mm2
        f_druck = 235             # N/mm2

        stresses = np.sign(E * strains) * np.minimum(f_druck, np.abs(E * strains))
        return stresses



class Steel_S355(Material):

    def __init__(self):

        gamma   = 78.5 * 10**(-6) # N/mm3
        E       = 210000          # N/mm2
        f_druck = 355             # N/mm2
        f_zug   = 355             # N/mm2

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (0, 0, 1, 0.5)
        self.name  = "Steel_S355" 
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        E       = 210000          # N/mm2
        f_druck = 355             # N/mm2
        stresses = np.sign(E * strains) * np.minimum(f_druck, np.abs(E * strains))
        return stresses


class Rebar_B500B(Material):

    def __init__(self):

        gamma   = 78.5 * 10**(-6) # N/mm3
        E       = 205000          # N/mm2
        f_druck = 435             # N/mm2
        f_zug   = 435             # N/mm2

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (0, 0.2, 1, 0.5)
        self.name  = "Rebar_B500B" 
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def get_stress_vectorized(strains):
        # Vectorized conditional logic using np.where¨
        E       = 205000          # N/mm2
        f_druck = 435             # N/mm2

        f_k  = 1.08 * f_druck
        e_s  = f_druck / E
        E_h  = (f_k - f_druck) / (0.05 - e_s)

        
        strains[strains < -0.05] = 0
        strains[strains >  0.05] = 0

        stresses = np.where(
            np.abs(strains) <= e_s,
            strains * E,
            np.sign(strains) * (f_druck + E_h * (np.abs(strains) - e_s))
        )
        return stresses

class Unknown(Material):

    def __init__(self):

        gamma   = 1
        E       = 1
        f_druck = 1
        f_zug   = 1 

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (1, 0, 0, 0.5)