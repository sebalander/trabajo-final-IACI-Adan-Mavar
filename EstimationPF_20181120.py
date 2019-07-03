#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "Claudio Adán, Esteban Mavar, Ulises Bussi and Sebastián Arroyo"
__license__ = "MIT"
__email__ = "sebastian.arroyo@unq.edu.ar"

Este código acompaña al Trabajo Final de Ingeniería en Automatización y Control Industrial de *Claudio Adán* y *Esteban Mavar*, titulado *"Detección y georreferenciación de vacantes de estacionamiento en la via pública"*, Universidad Nacional de Quilmes.

Ver sección 10.2 Procesamiento/Estimacion Georreferenciación.

MIT License

Copyright (c) 2019 Claudio Adán, Esteban Mavar, Ulises Bussi and Sebastián Arroyo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import smopy

def stdFilter(array):
    mean = np.mean(array)
    dosstd  = 1.6*np.std(array)
    aux= np.zeros_like(array)
    for i in range(len(array)):
        if np.abs(array[i]-mean) > dosstd :
            aux[i] = array[i-1]
        else:
            aux[i] = array[i]
    return aux


def lpFil(array):
    aux = (array+np.concatenate(([0],array[:-1])))/2
    return aux

def a02pi(array):
    return  (array + np.pi) % (2 * np.pi ) - np.pi


#%%


file = "20-Nov-2018_203616.csv"


columNames = ["lat", "lon", "year", "month", "day", "hour", "minute", "second",
              "speGPS", "nSats", "HDOP", "gyro"]
[columNames.append("USfront%d"%i) for i in range(8)]
[columNames.append("USback%d"%i) for i in range(8)]
[columNames.append("ENCleft%d"%i) for i in range(6)]
[columNames.append("ENCright%d"%i) for i in range(6)]
columNames.append("speOBD")
columNames.append("loopT")
conversion = 0.10854
# %% 
"""HAY UN PROBLEMA CON LOS DATOS DESPUÉS DE ESTOS EN LA FORMA EN LA QUE CREA EL TIEMPO"""
#cut = 751 
skip = 0
cut = 1400
# %%
data = pd.read_csv(file, skiprows=skip, header=None, names=columNames)
# compilo la fecha
data['datetime'] = pd.to_datetime(data[["year", "month", "day", "hour", "minute", "second"]])
# la convierto a segundos
data['time'] = (data.datetime - data.datetime[0]) / pd.Timedelta('1s')

try:
    map = np.load('map050918.npy').item()
except:
    margen = 0
    latMin = data.lat.min() - margen
    latMax = data.lat.max() + margen
    lonMin = data.lon.min() - margen
    lonMax = data.lon.max() + margen
    coordsMap = (latMin, lonMin, latMax, lonMax)    
    map = smopy.Map(coordsMap, margin=0)
    np.save('map050918.npy',map)

#matrices de ultrasound
matUSfront = np.array([data.USfront0, data.USfront1, data.USfront2, 
                       data.USfront3, data.USfront4, data.USfront5, 
                       data.USfront6, data.USfront7] )
matUSback  = np.array([data.USback0, data.USback1, data.USback2, 
                       data.USback3, data.USback4, data.USback5, 
                       data.USback6, data.USback7] )
#convierto en vector y asigno tiempos
timeUS  = np.arange(data.time[0],data.time[cut-1-skip],1/8)
USfront = np.concatenate(matUSfront.T)[:len(timeUS)]
USback  = np.concatenate(matUSback.T)[:len(timeUS)]

#matrices de encoders
matEncleft = np.array([data.ENCleft0, data.ENCleft1, data.ENCleft2, 
                       data.ENCleft3, data.ENCleft4, data.ENCleft5 ])
matEncright = np.array([data.ENCright0, data.ENCright1, data.ENCright2, 
                        data.ENCright3, data.ENCright4, data.ENCright5 ])
#convierto en vector y pongo en base de tiempo data.time
ENCleft = np.concatenate(matEncleft.T)
ENCright = np.concatenate(matEncright.T)
encR = np.zeros(len(data.time))
encL = np.zeros(len(data.time))
#si hay mas de una medición las promedio
for i in range(0,len(data.time)):
    auxR = ENCright[i*6:6*(i+1)]
    auxL = ENCleft [i*6:6*(i+1)]
    if (auxR>0).any():
        encR[i] = np.mean(auxR[auxR>0])    
    if (auxL>0).any():
        encL[i] = np.mean(auxL[auxL>0])


# %%
"""Eliminamos los Outliers con un filtro de std"""

L = 1.63 #estimativo?? en metros
cv = 1/3.6 #conversion de kmh a m/s
encr = stdFilter(encR)
encl = stdFilter(encL)
#plt.figure()
#plt.plot(ENCright[:751])
#plt.plot(ENCleft[:751])
##
#plt.figure()
#plt.plot(encL)
#plt.plot(encl)



offsetGyro =np.mean(data.gyro[:30])

Wenc = (encl-encr)*conversion*cv/L





#%%
dla = np.diff(data.lat)
dlo = np.diff(data.lon)


HeadGPS  = np.arctan2(dlo,dla) 
HeadEnc  = np.cumsum(Wenc)
HeadGyro = np.cumsum( (data.gyro-offsetGyro)*np.pi/180)

#HeadE02pi = a02pi(HeadEnc)
HeadG02pi = a02pi(HeadGyro)

#%%

from numpy import deg2rad, cos, sin, array, pi, eye, cumsum
from numpy import diag, concatenate, zeros, ones, abs, argmin
from numpy.random import randn, rand
from numpy.linalg import inv
Venc = (encr[:cut]+encl[:cut])*conversion/2

st=60
cut = 1400
# medición de entrada al sistema
cal =  deg2rad(data.gyro[:60]).mean()
y= np.array([data.speGPS[st:cut],data.speOBD[st:cut],Venc[st:cut],
             -(deg2rad(data.gyro[st:cut])-cal), Wenc[st:cut]]).T
y1= np.copy(y)
y1[:,1] *= (y[:,1]<500)
y1.T[0:3] /= 3.6

n = len(y1)



subm = 8
sts=st*subm
cuts= cut*subm
q = np.zeros((n*subm,5))
for i in range(5):
    q[:,i] = np.interp(np.arange(0,n,1/subm),np.arange(n),y1.T[i])
y1 = q


dt=1/subm
n = len(y1)
nPart = 700
nEst  = 3
nMeds = 2
x  = zeros((n,nPart,nEst))
xp = zeros((n,nPart,nEst))
wn = zeros([n,nPart]) 


stdMed = 2*(deg2rad(2.8*1.3e-5)*array([1,1]))
stdEst = 1e-1*array([deg2rad(1.3e-5) , deg2rad(1.3e-5) , deg2rad(10)] )
stdInic = array([5*deg2rad(1.3e-5) , 5*deg2rad(1.3e-5) , deg2rad(90)] )
stdIn = array([2 , 5 , 1 , deg2rad(10), deg2rad(10)])


L = diag(1/stdMed**2)


z= deg2rad(array([data.lat[st:cut], data.lon[st:cut]])).T

hd = array(data.HDOP[st:cut])

xp[0] = concatenate((z[10],[deg2rad(90)])) + randn(nPart,3)*stdInic





#geodetic
Lat0=z[0][0]
a = 6378.1370#Equatorial radius in km
b = 6356.7523#Polar radius in km
a_cos_cuad = (a*cos(Lat0))**2
b_sin_cuad = (b*sin(Lat0))**2
Rm = (a*b)**2/np.power(a_cos_cuad+b_sin_cuad,3/2) *1000#in meters
Rn = a**2 / np.sqrt(a_cos_cuad+b_sin_cuad)*1000#in meters
  

G = array([[2/8, 1/8, 5/8, 0  , 0],
           [0  , 0  , 0  , 29/30, 1/30]])
# G*y = u    

H = array([[1 , 0 , 0 ],
           [0 , 1 , 0 ] ])


cl = cos(Lat0)#de la lat
sl = sin(Lat0)#de la lat
clRn = cl*Rn    
zp = zeros(nPart)
op = ones(nPart)


def Prior(Xpost,inp,G,stdEst,stdIn,dt):
    [Vdt,Wdt] = G.dot(inp)*dt
    ct = cos(Xpost.T[2])#de theta
    st = sin(Xpost.T[2])#de theta
    deltaX =array([Vdt * ct/Rm ,
                   Vdt * st/(clRn),
                   Wdt * op]).T

    Fu = array([[dt * ct/Rm      , zp],
                [dt * st/(clRn)  , zp],
                [   zp           , dt*op] ]).T
    inpNoise = (Fu*(G.dot(stdIn)*randn(nPart,
                2)).reshape(nPart,-1,1)).sum(1)
    
    modNoise =  randn(nPart,nEst)*stdEst
    Xpred = Xpost + deltaX + inpNoise + modNoise
    
    return Xpred#,Cxprior


def Post(Xpred,med,H,stdMed,dt,L):
    medPred  = Xpred.dot(H.T)
    inn    = med-medPred
    w   = np.exp(-(inn.reshape(nPart,nMeds,1)*L.reshape(1,nMeds,nMeds)*\
                   inn.reshape(nPart,1,nMeds)).sum(axis=(1,2))) +1e-17
    wn = w/w.sum()
    Wcum = cumsum(wn)
    r = rand(nPart)
    d = abs(Wcum.reshape(1,nPart)-r.reshape(nPart,1))
    partSel = np.argmin(d,axis=1)
    Xpost  = Xpred[partSel]
    return Xpost



predx = []
predx.append(xp[0])

for i in range(n-1):
    xa  = Prior(xp[i],y1[i],G,stdEst,stdIn,dt)
    predx.append(xa)
    if not i%subm:
        j=int(i/8)-1
        stdMed1=stdMed*hd[j]/152
        lh = diag(1/stdMed1**2)
        xp[i+1]= Post(xa,z[j],H,stdMed1,dt,lh)
    else:
        xp[i+1] = xa
   
xparr= array(predx)






xm = xp.mean(1).T
sxm = xp.std(1)
t = np.arange(n)/8+1


#%%
y= np.array([data.speGPS[st:cut],data.speOBD[st:cut],Venc[st:cut],
             -(deg2rad(data.gyro[st:cut])-cal), Wenc[st:cut]]).T
a=100
b=250
plt.figure()
plt.plot(data.time[a:b], Venc[a:b],label='Velocidad Encoder')
plt.plot(data.time[a:b],data.speGPS[a:b],label='Velocidad GPS')
plt.plot(data.time[a:b],data.speOBD[a:b],label='Velocidad OBD')
plt.xlim((a,b))
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (km/h)')
plt.title('Comparación de velocidad entre los sensores')
plt.legend()

#%%
plt.figure()
aa = 8*100
bb = 8*120

plt.plot(timeUS[aa:bb],USback[aa:bb],label='Ultrasonido Trasero')
plt.plot(timeUS[aa:bb],USfront[aa:bb],label='Ultrasonido Delantero')
plt.xlim((aa/8,bb/8))
plt.xlabel('Tiempo (s)')
plt.ylabel('distancia (cm)')
plt.title('Comparación de UltraSonidos')
plt.legend()

#%%

def plots():
    d =640
    e = 10
    ds = int(d/subm)
    plt.figure()
    plt.subplot(311)
    plt.plot(t[::subm],z.T[0,:],label='lat GPS')
    plt.plot(t,xm[0],label='lat posterior')
    plt.plot(t[::d],xp[::d,::e,0],'+k')
    plt.plot(t[0],xm[0,0],'+k',label='disp')
    plt.legend()
    plt.subplot(312)
    plt.plot(t[::subm],z.T[1,:],label='lon GPS')
    plt.plot(t,xm[1],label='lon posterior')
    plt.plot(t[::d],xp[::d,::e,1],'+k')
    plt.plot(t[0],xm[1,0],'+k',label='disp')
    plt.legend()
    plt.subplot(313)
    plt.plot(t[::subm],np.unwrap(HeadGPS[st:cut]))
    plt.plot(t,xm[2])
    plt.plot(t[:-1:d],xp[::d,::e,2],'+k')
    
    plt.figure()
    plt.plot(z.T[1],z.T[0],label='tray gps')
    plt.plot(xm[1],xm[0],label='tray estimacion')
    plt.plot(xp[::d,::e,1],xp[::d,::e,0],'+k',alpha=0.15)
    plt.legend()
    
    plt.figure()
    plt.subplot(311)
    plt.plot(t[::subm],xm[0,::subm]-z.T[0,:],label='lat error')
    plt.plot(t[::subm*ds],(xp.T[0,:,::subm]-z.T[0].reshape(1,-1)).T[::ds,::e],'+k')
    plt.plot(t[0],0,'+k',label='disp')
    plt.legend()
    plt.subplot(312)
    plt.plot(t[::subm],xm[1,::subm]-z.T[1,:],label='lon error')
    plt.plot(t[::subm*ds],(xp.T[1,:,::subm]-z.T[1].reshape(1,-1)).T[::ds,::e],'+k')
    plt.plot(t[0],0,'+k',label='disp')
    plt.legend()
    plt.subplot(313)
    plt.plot(t[::subm],xm[2,::subm]-np.unwrap(HeadGPS[st:cut]),label='Head error')
    plt.plot(t[::subm*ds],(xp.T[2,:,::subm]-np.unwrap(HeadGPS[st:cut])).T[::ds,::e],'+k')
    plt.plot(t[0],0,'+k',label='disp')
    plt.legend()
    
    #Trayectoria
    (lae,loe) =map.to_pixels(np.rad2deg(xm[0]),np.rad2deg(xm[1]))
    (lap,lop) =map.to_pixels(np.array(data.lat),np.array(data.lon))
    plt.figure()
    plt.title('mapa y tray')
    plt.imshow(map.to_numpy())
    plt.plot(lap,lop,label='gps')
    plt.plot(lae,loe,label='est')

#
#
# %%
    
#USfront[::8].shape
def R_car_world(theta):
    cth = cos(theta)
    sth = sin(theta)
    zp  = zeros(len(theta))
    op  = ones(len(theta))
    return  array([[ -sth ,  cth , zp ],
                   [  cth , sth , zp ],
                   [  zp  ,  zp  , op ]]) 

"""Agrego información de cuando se ven los datos en la primer pasada:"""
#los indices son del tiempo de ultrasonido, para la pasada 1
#que arranca despues de 60 segs, y el tiempo de us es 1/8 de seg
#data.datetime[94] =Timestamp('2018-11-20 20:38:07')
#que se traduce como el punto (94-60)*8= 272
#data.datetime[98] = Timestamp('2018-11-20 20:38:11')
#que se traduce como el punto (98-60)*8= 304
#data.datetime[104] = Timestamp('2018-11-20 20:38:17')
#que se traduce como el punto (104-60)*8= 352
#data.datetime[110] = Timestamp('2018-11-20 20:38:23')
#que se traduce como el punto (110-60)*8= 400
#data.datetime[116] =Timestamp('2018-11-20 20:38:29')
#que se traduce como el punto (116-60)*8= 448
#data.datetime[120] =Timestamp('2018-11-20 20:38:3')
#que se traduce como el punto (120-60)*8= 480

"""Agrego información de cuando se ven los datos en la seg pasada:"""
#los indices son del tiempo de ultrasonido, para la pasada 2
#que arranca despues de 60 segs, y el tiempo de us es 1/8 de seg
#data.datetime[292] =Timestamp('2018-11-20 20:41:25')
#que se traduce como el punto (292-60)*8= 1856
#data.datetime[295  ] = Timestamp('2018-11-20 20:41:28')
#que se traduce como el punto (295-60)*8= 1880
#data.datetime[304] = Timestamp('2018-11-20 20:41:37')
#que se traduce como el punto (304-60)*8= 1952
#data.datetime[309] = Timestamp('2018-11-20 20:41:42')
#que se traduce como el punto (309-60)*8= 1992
#data.datetime[313] =Timestamp('2018-11-20 20:41:46')
#que se traduce como el punto (313-60)*8= 2024

idxPas1 = np.array([272,304,352,400,448,480])
idxPas2 = np.array([1856,1880,1952,1992,2024])
def PropWithMean():
    
    dxf = -2.03/2#1.63/2
    dyf = 2.92
    dxb = -2.03/2#1.63/2
    dyb = -0.47
    
    T_car_front = array([[  0 ,  1 , dxf ],
                         [  -1 ,  0 , dyf ],
                         [  0 ,  0 ,  1  ]])
    
    T_car_back  = array([[  0 ,  1 , dxb ],
                         [ -1 ,  0 , dyb ],
                         [  0 ,  0 ,  1  ]])
    
    zus = zeros(USfront[8*st:].shape)
    ous = ones(USfront[8*st:].shape)
    
#    Uf = array([zus, USfront[8*(st+1):]/100, ous])
#    Ub = array([zus,  USback[8*(st+1):]/100, ous])
    Uf = array([zus, USfront[8*st:]/100, ous])
    Ub = array([zus,  USback[8*st:]/100, ous])
    Ufc = T_car_front.dot(Uf)
    Ubc = T_car_back.dot(Ub)
    
#    Ufw =(R_car_world(xm[2][:-subm])*(Ufc.reshape(3,1,-1))).sum(0)
#    Ubw =(R_car_world(xm[2][:-subm])*(Ubc.reshape(3,1,-1))).sum(0)
    Ufw =(R_car_world(xm[2])*(Ufc.reshape(3,1,-1))).sum(0)
    Ubw =(R_car_world(xm[2])*(Ubc.reshape(3,1,-1))).sum(0)
    
    deltaFront = array([Ufw[0]/(clRn),Ufw[1]/Rm])
    deltaBack  = array([Ubw[0]/(clRn),Ubw[1]/Rm])
    
    start  = 0#500
    stop   = 10720#800   #10720
    start1 = 7350  #1500
    stop1  = 7650  #2200
    
    lo0 = xm[1][start]*clRn
    la0 = xm[0][start]*Rm
    
    #
    plt.figure()
    plt.plot(xm[1][start:stop]*clRn-lo0,
             xm[0][start:stop]*Rm -la0,
             label='Tray pasada 1')
    
    plt.plot((xm[1][start:stop]+deltaFront[1][start:stop])*clRn-lo0,
             (xm[0][start:stop]+deltaFront[0][start:stop])*Rm-la0,
             '+',label='UsFront pasada 1')
    
    plt.plot((xm[1][start:stop]+deltaBack[1][start:stop])*clRn-lo0,
             (xm[0][start:stop]+deltaBack[0][start:stop])*Rm-la0,
             '+',label='UsBack pasada 1')
    plt.legend()
    
    
    #Trayectoria
    (lae,loe) =map.to_pixels(np.rad2deg(xm[0][start:stop]),
                             np.rad2deg(xm[1][start:stop]))
    (laf,lof) =map.to_pixels(np.rad2deg(xm[0][start:stop]+deltaFront[0][start:stop]),
                             np.rad2deg(xm[1][start:stop]+deltaFront[1][start:stop]))
    (lab,lob) =map.to_pixels(np.rad2deg(xm[0][start:stop]+deltaBack[0][start:stop]),
                             np.rad2deg(xm[1][start:stop]+deltaBack[1][start:stop]))
    
    plt.figure()
    plt.title('mapa y tray')
    plt.imshow(map.to_numpy())
    plt.plot(lae,loe,label='est')
    plt.plot(laf,lof,label='front')
    plt.plot(lab,lob,label='back')
    
    plt.figure()
    plt.plot(xm[1][start:stop]*clRn-lo0,
             xm[0][start:stop]*Rm-la0,
             label='Tray pasada 1')
    plt.plot(xm[1][start1:stop1]*clRn-lo0,
             xm[0][start1:stop1]*Rm-la0,
             label='Tray pasada 2')
    plt.plot((xm[1][start:stop]+deltaFront[1][start:stop])*clRn-lo0,
             (xm[0][start:stop]+deltaFront[0][start:stop])*Rm-la0,
             label='UsFront pasada 1')
    plt.plot((xm[1][start1:stop1]+deltaFront[1][start1:stop1])*clRn-lo0,
             (xm[0][start1:stop1]+deltaFront[0][start1:stop1])*Rm-la0,
             label='UsFront pasada 2')
    plt.plot(xp.T[1,:,start:stop:100]*clRn-lo0,
             xp.T[0,:,start:stop:100]*Rm-la0,
             'k+',alpha=.2)
    plt.legend()
    
    (lae,loe) =map.to_pixels(np.rad2deg(xm[0][start:stop]),
                             np.rad2deg(xm[1][start:stop]))
    (laf,lof) =map.to_pixels(np.rad2deg(xm[0][start:stop]+deltaFront[0][start:stop]),
                             np.rad2deg(xm[1][start:stop]+deltaFront[1][start:stop]))
    (lae1,loe1) =map.to_pixels(np.rad2deg(xm[0][start1:stop1]),
                               np.rad2deg(xm[1][start1:stop1]))
    (laf1,lof1) =map.to_pixels(np.rad2deg(xm[0][start1:stop1]+deltaFront[0][start1:stop1]),
                               np.rad2deg(xm[1][start1:stop1]+deltaFront[1][start1:stop1]))
    
    
    plt.figure()
    plt.title('mapa y tray')
    plt.imshow(map.to_numpy())
    plt.plot(lae,loe,label='est pasada1')
    plt.plot(lae1,loe1,label='est pasada2')
    plt.plot(laf,lof,label='pasada1')
    plt.plot(laf1,lof1,label='pasada2')
    plt.plot(lae[idxPas1],loe[idxPas1],'+k')
    plt.plot(lae1[idxPas2-start1],loe1[idxPas2-start1],'+k')

