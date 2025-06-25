import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

# Definir la resolución de la cuadrícula HEALPix
nside = 2

# Obtener el número total de píxeles en la cuadrícula
npix = hp.nside2npix(nside)

# Crear un mapa de prueba con valores aleatorios
mapa = np.random.randn(npix)

# Dibuja el mapa en una proyección Mollweide usando Matplotlib
hp.mollview(mapa, title="Mapa de Calor Aleatorio", min=-2, max=2, unit="Unidad de Datos")

# Agregar líneas de latitud y longitud al mapa
hp.graticule()

# Agregar los números de índice de los píxeles en el mapa
for i in range(npix):
    # Obtener las coordenadas (θ, φ) del centro del píxel
    theta, phi = hp.pix2ang(nside, i)
    
    # Convertir las coordenadas de radianes a grados
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)
    
    # Dibujar el número de índice del píxel en su ubicación
    plt.text(phi_deg, theta_deg, str(i), horizontalalignment='center', verticalalignment='center', color='black', fontsize=100)

# Guardar la imagen como un archivo de imagen (por ejemplo, PNG)
plt.savefig('sky_map_with_index.png', dpi=300)

# Muestra la imagen utilizando Matplotlib
plt.show()

