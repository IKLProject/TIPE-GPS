import matplotlib.pyplot as plt
import math as m
import pandas as pd
import gpxpy
import numpy as np
import mplleaflet

#Ouverture fichier GPX

fh=open(r'C:\Users\SOS\Downloads\marathondeparis.gpx')
gpx_file = gpxpy.parse(fh)
segment = gpx_file.tracks[0].segments[0]

#Transformation du fichier en une table de coordonnées GPS

coords = pd.DataFrame([
        {'lat': p.latitude, 
         'lon': p.longitude, 
         'ele': p.elevation,
         'time': p.time} for p in segment.points])
coords.set_index('time', drop=True, inplace=True)

#Stockage des coordonnées GPS dans des listes

L_lat=coords['lat'].values
L_lon=coords['lon'].values
L_alti=coords['ele'].values

#Ajout de coordonnées NaNs pour les pertes de signal et les instants sans mesures

coords.index = np.round(coords.index.astype(np.int64), -9).astype('datetime64[ns]')
coords = coords.resample('1S').asfreq()

#Stockage des coordonnées GPS étendues dans des listes
    
L_lat_nan=coords['lat'].values
L_lon_nan=coords['lon'].values
L_alti_nan=coords['ele'].values

#Constantes pour les conversions entre coordonnées WGS 84 et coordonnées cartésiennes

a=6378137.
e=0.081819190842622
pi=3.14159265358979323846264338328

#Conversion WGS 84 (ellipsoïdal) -> Cartésien
  
def etc(lat,lon,alti):  
    
    global a,e,pi

    if np.isnan(lat):
        return np.nan, np.nan, np.nan
    lat,lon=lat*pi/180.,lon*pi/180.
    N=a/m.sqrt(1-e**2*m.sin(lat)**2)
    x=(N+alti)*m.cos(lat)*m.cos(lon)
    y=(N+alti)*m.cos(lat)*m.sin(lon)
    z=((1-e**2)*N+alti)*m.sin(lat)  
    return x,y,z

def conv_etc(t):
    
    n=len(t)
    L=[]
    for i in range(n):
        lat,lon,alti=t[i][0],t[i][1],t[i][2]
        x,y,z=etc(lat,lon,alti)
        L.append([x,y,z])
    return L

#Conversion cartésien -> WGS 84 (ellipsoïdal)        

def iter_lat_alti(N,lat,alti,p,z):
    
    global a,e,pi
    
    N=a/m.sqrt(1-e**2*m.sin(lat)**2)
    alti=p/m.cos(lat)-N
    lat=m.atan2(z,p*(1-e**2*N/(N+alti)))
    return N,lat,alti

def cte(x,y,z):
    
    global e,pi
    
    if np.isnan(x):
        return np.nan,np.nan,np.nan
    p=m.sqrt(x**2+y**2)
    lon=m.atan2(y,x)
    lat=m.atan2(z,p*(1-e**2))
    N,alti=0.,0.
    for i in range(10): #Nombre d'itérations nécessaires pour une latitude et une altitude précises
        N,lat,alti=iter_lat_alti(N,lat,alti,p,z)
    return lat*180/pi,lon*180/pi,alti
    
def conv_cte(t):
    
    n=len(t)
    L=[]
    for i in range(n):
        x,y,z=t[i][0],t[i][1],t[i][2]
        lat,lon,alti=cte(x,y,z)
        L.append([lat,lon,alti])
    return L

#Produit vectoriel et norme de vecteurs en 3D
    
def prod_vect(x0,y0,z0,x1,y1,z1):
    
    return y0*z1-y1*z0,z0*x1-z1*x0,x0*y1-x1*y0
    
def norme(x,y,z):
    
    return m.sqrt(x**2+y**2+z**2)

#Distance d'un point (x,y,z) à une droite de vecteur directeur (a,b,c) passant par le point (x0,y0,z0) 
    
def dist_3D(x,y,z,x0,y0,z0,a,b,c):
    
    t=prod_vect(x-x0,y-y0,z-z0,a,b,c)
    return norme(t[0],t[1],t[2])/norme(a,b,c)

#Concaténation de deux tableaux avec suppression du dernier élement du premier tableau
    
def concat(t1,t2):
    
    if t1[-1]==t2[0]:
        del t1[-1]
        return t1+t2
    return t1+t2

#Algorithme de Ramer-Douglas-Peucker en 3D avec epsilon la marge de tolérance de distance
  
def rdp_3D(t,epsilon):
    
    n=len(t)
    if n<=2:
        return t
    u,v=t[0],t[n-1]
    a,b,c=v[0]-u[0],v[1]-u[1],v[2]-u[2]
    maxdist,maxind=0,0
    for i in range(n):
        d_i=dist_3D(t[i][0],t[i][1],t[i][2],u[0],u[1],u[2],a,b,c)
        if d_i>maxdist:
            maxdist,maxind=d_i,i
    if maxdist<=epsilon:
        return [t[0],t[n-1]]
    else:
        return concat(rdp_3D(t[0:(maxind+1)],epsilon),rdp_3D(t[maxind:],epsilon))

#Ajout positions manquantes par régression linéaire

def long_nan_after(t,i):
    j=0
    while m.isnan(t[i+j]):
        j+=1
    return j

def fill_pos(t):
    n=len(t)
    i=0
    L=[]
    while i<n:
        j=long_nan_after(t,i)
        if j==0:
            L.append(t[i])
            i+=1
        else:
            x0,x1=t[i-1],t[i+j]
            for k in range(1,j+1):
                x=x0+k*(x1-x0)/(j+1)
                L.append(x)
            i+=j
    return L

#Calcul vitesses et accélérations

def fill_spd_accel(t):
    
    n=len(t)
    L=[0.]
    for i in range(1,n-1):
        L.append(t[i]-t[i-1])
    L.append(0.)
    return L

#Booléen de présence d'une valeur NaN dans une matrice
    
def test_nan(t):
    
    t=t.getA()
    n,p=t.shape
    for i in range(n):
        for j in range(p):
            if m.isnan(t[i][j]):
                return True
    return False

#Conversion liste de coordonnées WGS 84 -> positions + vitesses + accélérations cartésiennes
    
def convert_table(lat,lon,ele):
    n=len(lat)
    Lx,Ly,Lz=[],[],[]
    L_X,L_U=[],[]
    for i in range(n):
        x,y,z=etc(lat[i],lon[i],ele[i])
        Lx.append(x)
        Ly.append(y)
        Lz.append(z)
    Lx,Ly,Lz=fill_pos(Lx),fill_pos(Ly),fill_pos(Lz)
    Vx,Vy,Vz=fill_spd_accel(Lx),fill_spd_accel(Ly),fill_spd_accel(Lz)
    Ax,Ay,Az=fill_spd_accel(Vx),fill_spd_accel(Vy),fill_spd_accel(Vz)
    for i in range(n):
        L_X.append(np.matrix([[Lx[i]],[Ly[i]],[Lz[i]],[Vx[i]],[Vy[i]],[Vz[i]]]))
        L_U.append(np.matrix([[Ax[i]],[Ay[i]],[Az[i]]]))
    return L_X,L_U
          
#Listes des vecteurs d'état et des vecteurs de contrôle

L_X,L_U=convert_table(L_lat_nan,L_lon_nan,L_alti_nan)

#Intervalle entre deux mesures

dt=1

#Vecteur d'état et matrice de covariance initiales

X= L_X[0]

P=np.matrix([[10.,0.,0.,200.,0.,0.],[0.,10.,0.,0.,200.,0.],
             [0.,0.,10.,0.,0.,800.],[0.,0.,0.,20.,0.,0.],
             [0.,0.,0.,0.,20.,0.],[0.,0.,0.,0.,0.,20.]])

    
#Matrice de transition d'état    
    
F=np.matrix([[1., 0., 0., dt, 0., 0.],
             [0., 1., 0., 0., dt, 0.],
             [0., 0., 1., 0., 0., dt],
             [0., 0., 0., 1., 0., 0.],
             [0., 0., 0., 0., 1., 0.],
             [0., 0., 0., 0., 0., 1.]])

#Matrice associée au vecteur de contrôle

B = np.matrix([[0.5*dt**2,0.,0.],[0.,0.5*dt**2,0.],
               [0.,0.,0.5*dt**2],[dt,0.,0.],
               [0.,dt,0.],[0.,0.,dt]])

#Matrice d'observation
    
H=np.matrix([[1., 0., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0., 0.]])
 
#Matrice de covariance du bruit d'observation
    
R = np.matrix([[10., 0., 0.],
               [0., 10., 0.],
               [0., 0., 10.]])

#Matrice de covariance du bruit de processus

Q=np.zeros((6,6))

#Matrice identité

I = np.eye(6)

#Prédiction de l'état k+1 à partir de l'état k

def kf_predict(X,P,F,Q,B,U):
    
    if test_nan(U):
        X=F*X
    else:
        X=F*X+B*U
    P=F*P*F.T+Q
    return X,P
    
#Mise à jour de la prédiction avec la mesure de l'état k+1
    
def kf_update(X,P,Z,H,R):
    
    global I
    
    if test_nan(Z):
        return X,P
    Y=Z-H*X
    S=H*P*H.T+R
    K=(P*H.T)*np.linalg.pinv(S)
    X=X+K*Y
    P=(I-K*H)*P
    return X,P

#Implémentation filtre de Kalman
    
def filtre_Kalman(L_X,L_U):
    
    global dt,P,F,H,R,Q,B,U,I
    #Initialisation des listes de position, vitesse, gain de kalman,…
    n=len(L_X)
    Kalman_X=[L_X[0]]
    Kalman_P=[P]
    for k in range(n-1):
        Xk=Kalman_X[k]
        Pk=Kalman_P[k]
        Uk=L_U[k]
        Xk,Pk=kf_predict(Xk,Pk,F,Q,B,Uk)
        Z=L_X[k+1][0:3]
        Xk,Pk=kf_update(Xk,Pk,Z,H,R)
        Kalman_X.append(Xk)
        Kalman_P.append(Pk)
    return Kalman_X

#Transformation des résultats en coordonnées WGS 84

def convktx(t):
    L=[]
    for i in t:
        Li=i[0:3]
        Li=Li.getA()
        x,y,z=Li[0][0],Li[1][0],Li[2][0]
        L.append([x,y,z])
    return L

tab1,tab2=convert_table(L_lat_nan,L_lon_nan,L_alti_nan)

tab_kalman_cart=convktx(filtre_Kalman(tab1,tab2))

tab_kalman_GPS=conv_cte(tab_kalman_cart)

Kalman_lat,Kalman_lon=[],[]

for i in tab_kalman_GPS:
    Kalman_lat.append(i[0])
    Kalman_lon.append(i[1])

#Projection des résultats dans OpenStreetMap
    
fig=plt.figure()
plt.plot(Kalman_lon,Kalman_lat)
plt.plot(L_lon,L_lat)
mplleaflet.show(fig=fig)
