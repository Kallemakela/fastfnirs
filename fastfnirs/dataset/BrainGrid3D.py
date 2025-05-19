"""
3D grid for mapping locations to grid cells on a sphere
"""

# %%
#!%load_ext autoreload
#!%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from fastfnirs.dataset.BrainDataset import BrainDataset
from fastfnirs.utils import get_all_subjects

dataset_name = "bnci"
subjects = get_all_subjects(dataset_name)
subject = subjects[0]
bd = BrainDataset.load_data_2d(dataset_name, subjects=subjects)


class BrainGrid3D:
    def __init__(self, radius=1, spacing=0.4):
        self.radius = radius
        self.spacing = spacing
        self.grid_points, self.theta_phi_values = self.create_sphere_grid(
            radius, spacing
        )
        self.phis, self.thetas = zip(
            *self.theta_phi_values
        )  # Unzip to separate phi and theta lists

    def create_sphere_grid(self, radius, spacing):
        # Approximate number of points based on spacing
        circumference = 2 * np.pi * radius
        points_along_equator = int(circumference / spacing)
        dphi = 2 * np.pi / points_along_equator

        points = []
        theta_phi_values = []

        # Manually add the top point
        points.append((0, 0, radius))  # Top point at the pole
        theta_phi_values.append(
            (0, 0)
        )  # The top point has theta=0 and phi=0 for simplicity

        # Start theta from a small positive value to avoid multiple top points
        for phi in np.arange(0, 2 * np.pi, dphi):
            for theta in np.arange(
                spacing / radius, np.pi / 2, dphi
            ):  # Adjust theta start
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)

                points.append((x, y, z))
                theta_phi_values.append((theta, phi))

        return points, theta_phi_values

    def get_ix(self, x, y, z):
        # Calculate Euclidean distances from the input point to each grid point
        distances = [
            np.sqrt((gx - x) ** 2 + (gy - y) ** 2 + (gz - z) ** 2)
            for gx, gy, gz in self.grid_points
        ]
        nearest_index = np.argmin(
            distances
        )  # Index of the closest grid point based on Euclidean distance
        return nearest_index

    def get_coords(self, ix):
        if 0 <= ix < len(self.grid_points):
            return self.grid_points[ix]
        else:
            raise IndexError("Index out of range for grid points.")

    def get_ch2grid(self, chs):
        self.ch2grid = {}
        for ch in chs:
            x, y, z = ch["loc"][:3]
            ix = self.get_ix(x, y, z)
            ch_name = ch["ch_name"]
            self.ch2grid[ch_name] = ix
        return self.ch2grid

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i, (x, y, z) in enumerate(self.grid_points):
            ax.scatter(x, y, z, color="blue")  # Plot point
            ax.text(
                x, y, z, "%s" % (str(i)), size=10, zorder=1, color="k"
            )  # Show index

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    def plot2d(self):
        plt.figure(figsize=(8, 8))
        for i, (x, y, z) in enumerate(self.grid_points):  # Ignore z for top-down view
            plt.scatter(x, y, marker=".", color="blue")
            plt.text(
                x, y + 5e-2, str(i), color="red", fontsize=8, ha="center", va="center"
            )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Top-Down View of Sphere Grid with Indices")
        plt.axis("equal")  # Maintain aspect ratio
        plt.show()


brain_grid_3d = BrainGrid3D()
# brain_grid_3d.get_ch2grid(chs)
brain_grid_3d.plot2d()
# %%
# spherical meshgrid visualization of the top-down vie


def plot_meshgrid_with_cell_indices(bg, theta_steps=10, phi_steps=20):
    """
    Generate a meshgrid of points in spherical coordinates and plot them with their cell indices.
    :param theta_steps: Number of steps/divisions in theta.
    :param phi_steps: Number of steps/divisions in phi.
    """
    theta = np.linspace(0, np.pi / 2, theta_steps)
    phi = np.linspace(0, 2 * np.pi, phi_steps)

    plt.figure(figsize=(8, 8))

    cmap = plt.get_cmap("prism")
    ix_to_color = {
        ix: cmap(i / len(bg.grid_points))
        for i, ix in enumerate(range(len(bg.grid_points)))
    }

    # Generate and plot points from the meshgrid
    for t in tqdm(theta):
        for p in phi:
            x = bg.radius * np.sin(t) * np.cos(p)
            y = bg.radius * np.sin(t) * np.sin(p)
            z = bg.radius * np.cos(t)
            ix = bg.get_ix(x, y, z)
            color = ix_to_color[ix]
            plt.scatter(x, y, marker="o", color=color, alpha=0.5, edgecolors="none")

    for i, (x, y, z) in enumerate(bg.grid_points):
        color = ix_to_color[i]
        plt.scatter(x, y, marker="s", alpha=1, s=100, color=color, edgecolors="black")
        text_color = "black" if color[0] + color[1] + color[2] > 1.1 else "white"
        plt.text(x, y, str(i), color=text_color, fontsize=6, ha="center", va="center")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Top-Down View with Meshgrid Points and Indices")
    plt.axis("equal")


# Example usage
radius = 1
spacing = 0.3
brain_grid_3d = BrainGrid3D(radius, spacing)
plot_meshgrid_with_cell_indices(brain_grid_3d, theta_steps=50, phi_steps=50)
# plot_meshgrid_with_cell_indices(brain_grid_3d, theta_steps=100, phi_steps=200) # ~6min
# plt.savefig('sphere_grid_top_down.png', dpi=300)
plt.show()
# %%
