import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import healpy as hp

# Function to plot a sphere
def plot_sphere():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the sphere
    ax.plot_surface(x, y, z, color='b', alpha=0.2)
    
    # Set plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    # Set aspect ratio to be equal
    ax.set_box_aspect([1,1,1])
    
    plt.show()

# Function to plot selected pixels on the sphere with different colors for strip and ring
def plot_pixels(pixel_list_strip, pixel_list_ring):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.2)
    
    # Plot strip pixels
    for pixel in pixel_list_strip:
        for particle in pixel:
            theta = np.pi/2 - particle[0]  # Convert to colatitude
            phi = particle[1]
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            ax.scatter(x, y, z, color='r', marker='o', label='Strip')
    
    # Plot ring pixels
    for pixel in pixel_list_ring:
        for particle in pixel:
            theta = np.pi/2 - particle[0]  # Convert to colatitude
            phi = particle[1]
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            ax.scatter(x, y, z, color='g', marker='o', label='Ring')
    
    # Set plot limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    # Set aspect ratio to be equal
    ax.set_box_aspect([1,1,1])
    
    plt.legend()
    plt.show()

# Define functions to get strip and ring distributions
def get_strip_distribution(pixel_number, pixel_list, nside, num_pixels):
    theta, phi = hp.pix2ang(nside, pixel_number)
    vec = hp.pix2vec(nside, pixel_number)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    strip = hp.query_strip(nside, theta - num_pixels * d_theta, theta + num_pixels * d_theta)
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    strip = np.setdiff1d(strip, particle_ring)
    return [pixel_list[strip]]

def get_ring_distribution(pixel_number, pixel_list, nside, num_pixels):
    vec = hp.pix2vec(nside, pixel_number)
    d_theta = np.sqrt(hp.nside2pixarea(nside))
    particle_ring = hp.query_disc(nside, vec, num_pixels * d_theta)
    return [pixel_list[particle_ring]]

# Example usage
# Generate some random particle data for demonstration
np.random.seed(0)
num_particles = 1000
particle_data = np.random.rand(num_particles, 10)  # Random theta and phi values

# Plot the sphere
plot_sphere()

# Example usage of get_strip_distribution
strip_pixel = get_strip_distribution(0, particle_data, nside=16, num_pixels=4)
#plot_pixels([strip_pixel])

# Example usage of get_ring_distribution
ring_pixel = get_ring_distribution(0, particle_data, nside=16, num_pixels=4)
#plot_pixels([ring_pixel])

plot_pixels([strip_pixel], [ring_pixel])
