import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as tri
from matplotlib import cm, colors

from tabulate import tabulate

import element
import geometry
import mesh
import material

class stress_strain_analysis:
    def __init__(self, mesh, Nx = 0, My = 0):

        self.mesh   = mesh
        self.Nx     = Nx
        self.My     = My

        self.strains  = []
        self.stresses = []

        self.eps_x = 0
        self.xsi   = 0

    def set_strain_and_curvature(self, eps_x, xsi):
        self.eps_x = eps_x
        self.xsi   = xsi 

    def calculate_strains(self):
        
        self.strains = np.array([])

        for elem in self.mesh.elements:
            eps_normal = self.eps_x
            eps_cruv   = (elem.Cy - self.mesh.Cy) * self.xsi

            self.strains = np.append(self.strains, eps_normal + eps_cruv)

        return
    
    def calculate_stresses(self):

        self.stresses = np.array([])

        for i, elem in enumerate(self.mesh.elements):
            stress = elem.material.get_stress(self.strains[i])

            self.stresses = np.append(self.stresses, stress)
        
        return

    def get_section_forces(self):
        if len(self.stresses) == 0:
            raise ValueError("Stresses have not been calculated. Run calculate_stresses() first.")
        if len(self.strains) == 0:
            raise ValueError("Strains have not been calculated. Run calculate_strains() first.")

        N  = 0
        My = 0

        for i, elem in enumerate(self.mesh.elements):
            N  += elem.A * self.stresses[i]
            My += elem.A * self.stresses[i] * (elem.Cy - self.mesh.Cy)

        N  = N  / 1000         # kN
        My = My / 1000 / 1000 # kNm
        
        return N, My
    
    def find_strain_and_curvature(self, V):

        self.set_strain_and_curvature(V[0], V[1])
        self.calculate_strains()
        self.calculate_stresses()

        Nx, My = self.get_section_forces()

        Residual = (Nx - self.Nx)**2 + (My - self.My)**2

        return Residual


    def plot_strains(self):
        if len(self.strains) == 0:
            raise ValueError("Strains have not been calculated. Run calculate_strains() first.")

        # Normalize strains for color mapping
        max_strain = max(abs(min(self.strains)), abs(max(self.strains)))
        norm = colors.TwoSlopeNorm(vmin=-max_strain, vcenter=0, vmax=max_strain)
        cmap = cm.get_cmap('coolwarm')

        fig, ax = plt.subplots(figsize=(6, 6))
        for i, elem in enumerate(self.mesh.elements):
            x = elem.coords[:, 0]
            y = elem.coords[:, 1]
            poly = patches.Polygon(np.column_stack([x, y]),
                                   edgecolor='black',
                                   facecolor=cmap(norm(self.strains[i])),
                                   lw=0.3)
            ax.add_patch(poly)

        # Plotting nodes to improve visual correctness without marking them
        ax.plot(self.mesh.node_coords[:, 0],
                self.mesh.node_coords[:, 1],
                'o', markersize=0, color='black')

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Strain")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_frame_on(False)
        ax.set_title("Strain Visualization")
        ax.set_aspect('equal')
        plt.show()

    def plot_stresses(self):
        if len(self.stresses) == 0:
            raise ValueError("Stresses have not been calculated. Run calculate_stresses() first.")

        # Normalize strains for color mapping
        max_strain = max(abs(min(self.stresses)), abs(max(self.stresses)))
        norm = colors.TwoSlopeNorm(vmin=-max_strain, vcenter=0, vmax=max_strain)
        cmap = cm.get_cmap('coolwarm')

        fig, ax = plt.subplots(figsize=(6, 6))
        for i, elem in enumerate(self.mesh.elements):
            x = elem.coords[:, 0]
            y = elem.coords[:, 1]
            poly = patches.Polygon(np.column_stack([x, y]),
                                   edgecolor='black',
                                   facecolor=cmap(norm(self.stresses[i])),
                                   lw=0.3)
            ax.add_patch(poly)

        # Plotting nodes to improve visual correctness without marking them
        ax.plot(self.mesh.node_coords[:, 0],
                self.mesh.node_coords[:, 1],
                'o', markersize=0, color='black')

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Stress")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_frame_on(False)
        ax.set_title("Stress Visualization")
        ax.set_aspect('equal')
        plt.show()
