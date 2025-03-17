import numpy as np
from tabulate import tabulate

class Laminate:
    def __init__(self, layers):
        """
        layers: list of dictionaries, each containing:
            - 'E1': Young's modulus in fiber direction
            - 'E2': Young's modulus perpendicular to fiber direction
            - 'G12': Shear modulus
            - 'v12': Poisson's ratio
            - 'theta': fiber orientation in degrees
            - 'thickness': thickness of the ply
        """
        self.layers = layers
        self.n      = len(layers)
        self.h      = np.cumsum([0] + [l['thickness'] for l in layers]) - sum(l['thickness'] for l in layers) / 2
        self.A, self.B, self.D = self.compute_ABD_matrix()
        self.Ex, self.Ey, self.Gxy, self.vxy = self.compute_equivalent_properties()
    
    def Q_matrix(self, E1, E2, G12, v12):
        """Computes the reduced stiffness matrix Q for a single ply"""
        v21 = (E2 * v12) / E1  # Reciprocal Poisson's ratio
        Q11 = E1 / (1 - v12 * v21)
        Q22 = E2 / (1 - v12 * v21)
        Q12 = v12 * E2 / (1 - v12 * v21)
        Q66 = G12
        return np.array([[Q11, Q12,    0],
                         [Q12, Q22,    0],
                         [  0,   0,  Q66]])
    
    def transform_Q(self, Q, theta):
        """Transforms the stiffness matrix Q to the laminate coordinate system"""
        theta = np.radians(theta)
        c, s = np.cos(theta), np.sin(theta)
        T = np.array([[ c**2, s**2,       2*c*s],
                      [ s**2, c**2,      -2*c*s],
                      [-c*s ,  c*s, c**2 - s**2]])
        return T @ Q @ np.linalg.inv(T)
    
    def compute_ABD_matrix(self):
        """Computes the A, B, and D matrices of the laminate"""
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))
        
        for i, layer in enumerate(self.layers):
            Q = self.Q_matrix(layer['E1'], layer['E2'], layer['G12'], layer['v12'])
            Q_bar = self.transform_Q(Q, layer['theta'])
            h_k = self.h[i]
            h_k1 = self.h[i + 1]
            A += Q_bar * (h_k1 - h_k)
            B += (1/2) * Q_bar * (h_k1**2 - h_k**2)
            D += (1/3) * Q_bar * (h_k1**3 - h_k**3)
        
        return A, B, D
    
    def compute_equivalent_properties(self):
        """Computes the equivalent material properties of the laminate."""
        h_total = self.h[-1] - self.h[0]
        Ex  = self.A[0, 0] / h_total
        Ey  = self.A[1, 1] / h_total
        Gxy = self.A[2, 2] / h_total
        vxy = self.A[0, 1] / self.A[1, 1]
        return Ex, Ey, Gxy, vxy

    def get_ABD_matrix(self):
        return self.A, self.B, self.D
    
    def get_equivalent_properties(self):
        return self.Ex, self.Ey, self.Gxy, self.vxy

    def print_results(self):
        """Prints the ABD matrix and equivalent properties in tabular format."""
        headers = ["A Matrix", "B Matrix", "D Matrix"]
        data = [[np.array2string(self.A, formatter={'float_kind': lambda x: f'{x:.2e}'}),
                np.array2string(self.B, formatter={'float_kind': lambda x: f'{x:.2e}'}),
                np.array2string(self.D, formatter={'float_kind': lambda x: f'{x:.2e}'})]]
        print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

        properties = [
            ["Ex" , f"{self.Ex:.2e}"],
            ["Ey" , f"{self.Ey:.2e}"],
            ["Gxy", f"{self.Gxy:.2e}"],
            ["vxy", f"{self.vxy:.2e}"]
        ]
        print(tabulate(properties, headers=["Property", "Value"], tablefmt="fancy_grid"))


class LaminateLoadAnalysis:
    def __init__(self, laminate):
        self.laminate = laminate
    
    def apply_load(self, Nx, Ny, Nxy, Mx=0, My=0, Mxy=0):
        N = np.array([Nx, Ny, Nxy])
        M = np.array([Mx, My, Mxy])
        
        ABD = np.block([[self.laminate.A, self.laminate.B], [self.laminate.B, self.laminate.D]])
        
        loads = np.concatenate((N, M))
        
        midplane_strains_curvatures = np.linalg.solve(ABD, loads)
        midplane_strains = midplane_strains_curvatures[:3]
        midplane_curvatures = midplane_strains_curvatures[3:]
        
        return midplane_strains, midplane_curvatures

    def compute_ply_stresses_strains(self, midplane_strains, midplane_curvatures):
        ply_stresses = []
        ply_strains = []
        
        for i, layer in enumerate(self.laminate.layers):
            Q = self.laminate.Q_matrix(layer['E1'], layer['E2'], layer['G12'], layer['v12'])
            Q_bar = self.laminate.transform_Q(Q, layer['theta'])
            
            z = self.laminate.h[i]  # Abstand zur Mittelfläche
            
            strain = midplane_strains + z * midplane_curvatures
            stress = Q_bar @ strain
            
            ply_strains.append(strain)
            ply_stresses.append(stress)
        
        return np.array(ply_strains), np.array(ply_stresses)

    def print_ply_results(self, ply_strains, ply_stresses):
        strain_table = [[i + 1] + list(map(lambda x: f"{x:.2e}", ply_strains[i])) for i in range(len(ply_strains))]
        stress_table = [[i + 1] + list(map(lambda x: f"{x:.2e}", ply_stresses[i])) for i in range(len(ply_stresses))]
        
        print("\nDehnungen pro Schicht:")
        print(tabulate(strain_table, headers=["Schicht", "Exx", "Eyy", "Gxy"], tablefmt="fancy_grid"))
        
        print("\nSpannungen pro Schicht:")
        print(tabulate(stress_table, headers=["Schicht", "Sxx", "Syy", "Txy"], tablefmt="fancy_grid"))