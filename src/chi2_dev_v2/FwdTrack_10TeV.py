#program to make map at "Earth", actually at some radius near to Earth
# import proper libraries
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import random
import healpy as H
import numpy

# import data


from numpy  import *

nfile = 1 #The number of data files (in our case is just one file)
#min_radius = 1.1 #very very close to the origin
max_radius = 300.00 #max_radius The radius of the pixels of interest, this is in grids NOT AU

#coordinates of the center, e.g. where the Earth is located
xe = 460.5000 
ye = 256.5000
ze = 256.5000
#rc = sqrt((xc)**2+(yc)**2+(zc)**2) #radius at the center

#value of the magnetic field at max radius
bx = 1.791
by = -1.435
bz = 1.917
B = sqrt((bx)**2+(by)**2+(bz)**2)

#value for parallel component of the dipole (parallel to the magnetic field) 
dip = 1

data = loadtxt("R20_300_10TeV.txt") #checked syntax

pxe = data[:,3] #load x momentum data at Earth
pye = data[:,4] #load y momentum data
pze = data[:,5] #load z momentum data

xr = data[:,7] #load x trajectory data at max radius
yr = data[:,8] #load y trajectory data
zr = data[:,9] #load z trajectory data

pxr = data[:,10] #load x momentum data at max radius
pyr = data[:,11] #load y momentum data
pzr = data[:,12] #load z momentum data

size = xr.shape #get number of particles
size = size[0] #contd 
iPart=size

#preparing the arrays that will be used
F = zeros((iPart,1))
phitote = zeros((iPart,1))
thetatote = zeros((iPart,1))
vec_celm = zeros((iPart,1))
phitotr = zeros((iPart,1))
thetatotr = zeros((iPart,1))
w = zeros((iPart,1))

counter = 0
for idP in range (0,size):
    print(idP)
    counter = counter + 1
    
    pxri = pxr[idP]
    pyri = pyr[idP]
    pzri = pzr[idP]
    pri = sqrt((pxri)**2+(pyri)**2+(pzri)**2)

    
#    print 'Particles counted so far:' , str(counter)
    
#dipole distribution weight
    F[idP] = (dip/(B*pri))*(bx*pxri + by*pyri + bz*pzri )
    
#now at Earth
#    pxei = pxe[idP]
#    pyei = pye[idP]
#    pzei = pze[idP]
#    pei = sqrt((pxei)**2+(pyei)**2+(pzei)**2)

#    phie= arctan2((pyei),(pxei))
#    phitote[idP] = phie
    
#    thetae = arccos((pzei)/(pei))
#    thetatote[idP] = thetae

  
    xri = xr[idP]
    yri = yr[idP]
    zri = zr[idP]
    rri = sqrt((xri-xe)**2+(yri-ye)**2+(zri-ze)**2)

    phir= arctan2((yri-ye),(xri-xe))
    phitotr[idP] = phir
    
    thetar = arccos((zri-ze)/(rri))
    thetatotr[idP] = thetar

    if phitotr[idP] > 0:
        w1=(1.346/0.2361)
        w[idP] = w1
    elif phitotr[idP] < 0:
        w[idP] = 1
        


    
    
#import graph libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as n

npix=0
map=0
#map1=0
pixn=0

nside = 16 # resolution, number pixels in along side of healpix basis pixel (basis is nside=1)                   
npix = H.nside2npix(nside)# total number of pixels in map
map = n.zeros(npix, dtype=n.double)
#map1 = n.zeros(npix, dtype=n.double)
#Npix. Denotes the total number of pixels. Npix=12Nside^2
print("this is the number of pixels:")
print(npix)
#matrix = n.zeros(shape=(npix,npix))



#change of coordinates:
#From NICKS (Our in MAPLE file) to Ecliptic (HAE)
#Using equation 17 from MAPLE file

NickEcl= n.matrix([[-0.202372670869508942, 0.971639226673224665 , 0.122321361599999998],[-0.979292047083733075, -0.200058547149551208, -0.0310429431300000003],[-0.00569110735590557925, -0.126070579934110472, 0.992004949699999972]])



for iP in range(0,iPart):
    pxei = pxe[iP]
    pyei = pye[iP]
    pzei = pze[iP]
    pei = n.sqrt((pxei)**2+(pyei)**2+(pzei)**2)

#rotate by matrix
    vectorEcl = H.rotator.rotateVector(NickEcl,pxei,pyei,pzei)

#change from ecliptic to equatorial:
#using healpy functions

  
# INFO r = Rotator(coord=['G','E'])  # Transforms galactic to ecliptic coordinates
    r = H.Rotator(coord=['E','C'])
    vec_cel = r(vectorEcl)


#assuming the vectorEcl is [x,y,z]
    vec_celm[iP] = n.sqrt((vec_cel[0])**2+(vec_cel[1])**2+(vec_cel[2])**2)    

    phie= n.arctan2((vec_cel[1]),(vec_cel[0]))
    phitote[iP] = phie
    
    thetae = n.arccos((vec_cel[2])/(vec_celm[iP]))
    thetatote[iP] = thetae

#    pixni = H.ang2pix(nside, thetatot1[iP], phitot1[iP])# convert theta and phi to pixels for min_radius
    pixn = H.ang2pix(nside, thetatote[iP], phitote[iP]) # convert theta and phi to pixels for max_radius
#    particle_pixel[iP,0] = shell_n[iP] #create array recording particle number
#    particle_pixel[iP,1] =pixn #create array recording index number
#    map1[pixni] += 1
    map[pixn] += 1*F[iP]*w[iP]            
#    matrix[pixni,pixn]+=1

#AU = (2*max_radius-1)*10 #convert radius to AU units
#AU = n.around(AU)
#phirange = n.linspace(-n.pi, n.pi,32)
#thetarange = n.linspace(0, n.pi,32)
#print phirange
#n.save('particle_map_radius'+str(AU), matrix) #save particle,pixel array

H.mollview(map, title='Distribution of 10 TeV Cosmic Rays at Earth Forward Tracking Cel R=20', unit='Cosmic Ray Permeation Count')
H.graticule()
#AU = int(AU)
plt.savefig('Map_10TeV_at_Earth_FwdTrack_Celestial_Dipole_R20')

#To plot counts vs pixel number in an specific area
#plt.plot(map)
#plt.axis([1500, 3000, 0, 1200])
#plt.savefig('Map_zones2and2_radius_' +str(AU))

plt.clf()








