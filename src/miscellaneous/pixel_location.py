import healpy as hp
import numpy as np

# Definir la resolución de la cuadrícula HEALPix
nside = 2
# Obtener el número total de píxeles en la cuadrícula
npix = hp.nside2npix(nside)

# Crear un vector para almacenar el número del píxel y sus coordenadas en grados
pixel_locations = []

# Iterar sobre todos los píxeles
for i in range(npix):
    # Obtener las coordenadas (θ, φ) del centro del píxel
    theta, phi = hp.pix2ang(nside, i)
    
    # Convertir las coordenadas de radianes a grados y redondear a dos decimales
    theta_deg = round(np.degrees(theta), 2)
    phi_deg = round(np.degrees(phi), 2)
    
    # Almacenar el número del píxel y sus coordenadas en grados en el vector
    pixel_locations.append((i, theta_deg, phi_deg))

# Convertir el vector en un array numpy para facilitar su manipulación si es necesario
pixel_locations = np.array(pixel_locations)

# Imprimir el vector de ubicaciones de los píxeles
for pixel_location in pixel_locations:
    print("Pixel:", pixel_location[0], " - Theta:", pixel_location[1], " - Phi:", pixel_location[2])

