
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from copy import deepcopy
from itertools import combinations

class SphereCoord:
    def __init__(self, radius, x=None, y=None, z=None, theta=None, phi=None):
        self.radius = radius # radius of the sphere that the point is on
        if x is not None and y is not None and z is not None:
            self.x, self.y, self.z = x, y, z
            self._cartesian_to_spherical()
        elif theta is not None and phi is not None:
            self.theta, self.phi = theta, phi
            self.r = radius
            self._spherical_to_cartesian()
        else:
            raise ValueError("Insufficient coordinates provided")
        
        self._validate_on_sphere()
    
    def _cartesian_to_spherical(self):
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.theta = np.arccos(self.z / self.r)
        self.phi = np.arctan2(self.y, self.x)
    
    def _spherical_to_cartesian(self):
        self.x = self.radius * np.sin(self.theta) * np.cos(self.phi)
        self.y = self.radius * np.sin(self.theta) * np.sin(self.phi)
        self.z = self.radius * np.cos(self.theta)
    
    def _validate_on_sphere(self):
        if not np.isclose(self.r, self.radius):
            print("Point is not on the sphere. Consider scaling.")

    @property
    def cartesian(self):
        return self.x, self.y, self.z
    
    @property
    def spherical(self):
        return self.r, self.theta, self.phi
    
    def __repr__(self):
        return f"SC({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"
    
    def scale_to_sphere(self):
        self._spherical_to_cartesian()  # Recalculate Cartesian coordinates
        self._cartesian_to_spherical()  # Update spherical coordinates to ensure consistency
    
    def add_to_theta_phi(self, dtheta, dphi):
        self.theta += dtheta
        self.phi += dphi
        self._spherical_to_cartesian()
    
    def add_to_xyz(self, dx, dy, dz):
        self.x += dx
        self.y += dy
        self.z += dz
        self._cartesian_to_spherical()
        self.scale_to_sphere()  # Ensure the point remains on the sphere
    
    def sphere_distance(self, other):
        return np.arccos(
            np.sin(self.theta) * np.sin(other.theta) * np.cos(self.phi - other.phi) + np.cos(self.theta) * np.cos(other.theta)
        ) * self.radius
    
    def point_at_distance(self, distance, direction):
        """
        Generate a new point on the sphere at a given distance from this point.
        
        :param distance: The great circle distance on the sphere surface
        :param direction: tuple (dtheta, dphi) for a specific direction
        :return: A new SphereCoord object
        """
        dtheta, dphi = direction

        # Calculate the new theta and phi
        new_theta = np.arccos(np.cos(distance / self.radius) * np.cos(self.theta) + 
                                np.sin(distance / self.radius) * np.sin(self.theta) * np.cos(dtheta))
        
        new_phi = self.phi + np.arctan2(np.sin(dtheta) * np.sin(distance / self.radius) * np.sin(self.theta),
                                        np.cos(distance / self.radius) - np.cos(self.theta) * np.cos(new_theta))

        # Normalize phi to be within [0, 2π)
        new_phi = (new_phi + 2*np.pi) % (2*np.pi)

        # Create and return the new point
        return SphereCoord(self.radius, theta=new_theta, phi=new_phi)


class SphereProjector:
    def __init__(self, radius):
        self.radius = radius
    
    def rotate_around_y(self, coords, theta):
        rotation_matrix = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        return np.dot(coords, rotation_matrix.T)
    
    def rotate_around_z(self, coords, theta):
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return np.dot(coords, rotation_matrix.T)
    
    def grid_to_sphere(self, grid_coords):
        grid_coords = np.array(grid_coords)[:,:2]
        angular_displacement = 1 / self.radius
        projected_coords = []
        for x, y in grid_coords:
            theta = np.pi / 2 - (y * angular_displacement)  # Polar angle, from the top
            phi = x * angular_displacement  # Azimuthal angle
            
            # Convert spherical coordinates to Cartesian coordinates
            x = self.radius * np.sin(theta) * np.cos(phi)
            y = self.radius * np.sin(theta) * np.sin(phi)
            z = self.radius * np.cos(theta)
            
            projected_coords.append([x, y, z])
        
        projected_coords = np.array(projected_coords)
        theta = -np.pi / 2
        projected_coords = self.rotate_around_y(projected_coords, theta)
        theta = -np.pi / 2
        projected_coords = self.rotate_around_z(projected_coords, theta)
        return projected_coords

    def sphere_distance_xyz(self, p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dot = x1 * x2 + y1 * y2 + z1 * z2
        cos_theta = max(min(dot / (self.radius ** 2), 1), -1)
        angle = math.acos(cos_theta)
        return angle * self.radius

    def distance_matrix(self, locs):
        if type(locs[0]) == SphereCoord:
            locs = [p.cartesian for p in locs]
        dists = np.zeros((len(locs), len(locs)))
        for i, loc1 in enumerate(locs):
            for j, loc2 in enumerate(locs):
                dists[i, j] = self.sphere_distance_xyz(loc1, loc2)
        return dists

    def plot_projected(self, coords):
        if type(coords[0]) == SphereCoord:
            coords = [p.cartesian for p in coords]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        u, v = np.mgrid[0:2*np.pi:50j, 0:1.25*np.pi/2:25j]
        x_half_sphere = self.radius * np.cos(u) * np.sin(v)
        y_half_sphere = self.radius * np.sin(u) * np.sin(v)
        z_half_sphere = self.radius * np.cos(v)
        ax.plot_surface(x_half_sphere, y_half_sphere, z_half_sphere, color="lightblue", alpha=0.4)
        
        for ci, (x, y, z) in enumerate(coords):
            ax.text(x, y, z, f"{ci}", color='black', fontsize=12)
            ax.scatter(x, y, z, color="red", s=50, alpha=1)

        ax.view_init(elev=20, azim=45)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_zlim(-self.radius, self.radius)
        return fig


def create_neighbor_mask(grid_size):

    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    
    # Initialize the mask with False values
    neighbor_mask = np.zeros((grid_size[0] * grid_size[1], grid_size[0] * grid_size[1]), dtype=bool)
    
    # Iterate over each cell in the grid
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            # Calculate the flattened index for the current cell
            i = y * grid_size[0] + x
            
            # Check for each of the four possible neighbors
            # and update the mask accordingly
            
            # Left neighbor
            if x > 0:
                neighbor_mask[i, i-1] = True
                
            # Right neighbor
            if x < grid_size[0] - 1:
                neighbor_mask[i, i+1] = True
                
            # Up neighbor
            if y > 0:
                neighbor_mask[i, i - grid_size[0]] = True
                
            # Down neighbor
            if y < grid_size[1] - 1:
                neighbor_mask[i, i+grid_size[0]] = True
                
    return neighbor_mask

def optimize_circle_coords(coords, neighbor_mask, target_distance=None):

    radius = coords[0].radius
    projector = SphereProjector(radius)
    
    if target_distance is None:
        neighbors = np.where(neighbor_mask)
        neighbor_distances = []
        for i, j in zip(*neighbors):
            neighbor_distances.append(np.round(coords[i].sphere_distance(coords[j]), 3))
        u, c = np.unique(neighbor_distances, return_counts=True)
        target_distance = u[np.argmax(c)]

    def objective(offsets, projected_coords, neighbor_mask):
        offsetted_coords = [deepcopy(p) for p in projected_coords]
        for i, (dtheta, dphi) in enumerate(offsets.reshape((-1, 2))):
            offsetted_coords[i].add_to_theta_phi(dtheta, dphi)
        distance_matrix = projector.distance_matrix(offsetted_coords)
        diff = (distance_matrix[neighbor_mask] - target_distance)
        obj = np.sum(diff ** 2)
        return obj

    initial_offsets = np.zeros((len(coords), 2)).flatten()
    result = minimize(
        objective, 
        initial_offsets, 
        args=(coords, neighbor_mask),
        method='L-BFGS-B'  # A commonly used optimization algorithm that handles bounds well
    )

    # Update theta_phi_offsets with the optimized values
    theta_phi_offsets_optimized = result.x.reshape((-1, 2))

    offsetted_coords = [deepcopy(p) for p in coords]
    for i, (dtheta, dphi) in enumerate(theta_phi_offsets_optimized):
        offsetted_coords[i].add_to_theta_phi(dtheta, dphi)
    
    return offsetted_coords

#%%
# radius = 0.08
# projector = SphereProjector(radius)
# grid_size = 3
# dist_between_points = 0.025
# grid_coords = np.array([(x, y) for y in range(grid_size) for x in range(grid_size)]) * dist_between_points + 0.02
# neighbor_mask = create_neighbor_mask(grid_size)
# projected_coords_circle = projector.grid_to_sphere(grid_coords)
# projected_coords_circle = [SphereCoord(radius, x=x, y=y, z=z) for x, y, z in projected_coords_circle]
# distance_matrix_result = projector.distance_matrix(projected_coords_circle)

# print(distance_matrix_result[neighbor_mask])
# print(np.sum((distance_matrix_result[neighbor_mask] - dist_between_points) ** 2))

# projected_coords_circle = optimize_circle_coords(projected_coords_circle, target_distance=dist_between_points)
# distance_matrix_result = projector.distance_matrix(projected_coords_circle)

# print(distance_matrix_result[neighbor_mask])
# print(np.sum((distance_matrix_result[neighbor_mask] - dist_between_points) ** 2))

# projected_coords_circle_mirror = [SphereCoord(radius, x=-c.x, y=c.y, z=c.z) for c in projected_coords_circle]

# coords = projected_coords_circle + projected_coords_circle_mirror

# fig = projector.plot_projected(coords)
# fig.gca().view_init(elev=30, azim=90)
# plt.show()
# %%
