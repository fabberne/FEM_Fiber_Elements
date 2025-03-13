import numpy as np

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
    
    def get_stress(self, strain):

        stress = self.E * strain

        if stress > self.f_druck:
            stress = self.f_druck
        elif stress < -self.f_zug:
            stress = 0

        return stress


class Steel_S235(Material):

    def __init__(self):

        gamma   = 78.5 * 10**(-6) # N/mm3
        E       = 210000          # N/mm2
        f_druck = 235             # N/mm2
        f_zug   = 235             # N/mm2

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (0, 0, 1, 0.5)

    def get_stress(self, strain):

        stress = np.sign(self.E * strain) * min(self.f_druck, abs(self.E * strain))

        return stress



class Steel_S355(Material):

    def __init__(self):

        gamma   = 78.5 * 10**(-6) # N/mm3
        E       = 210000          # N/mm2
        f_druck = 355             # N/mm2
        f_zug   = 355             # N/mm2

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (0, 0, 1, 0.5)
    
    def get_stress(self, strain):

        stress = np.sign(self.E * strain) * min(self.f_druck, abs(self.E * strain))

        return stress


class Rebar_B500B(Material):

    def __init__(self):

        gamma   = 78.5 * 10**(-6) # N/mm3
        E       = 205000          # N/mm2
        f_druck = 435             # N/mm2
        f_zug   = 435             # N/mm2

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (0, 0.2, 1, 0.5)

class Unknown(Material):

    def __init__(self):

        gamma   = 1
        E       = 1
        f_druck = 1
        f_zug   = 1 

        super().__init__(gamma, E, f_druck, f_zug)

        self.color = (1, 0, 0, 0.5)