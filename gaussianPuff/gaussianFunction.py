#gaussianFunction.py
import numpy as np
from scipy.special import erfcinv as erfcinv
from sigmaCalculation import calc_sigmas 
from numpy import sqrt

def gauss_func(Q,u,dir1,x,y,z,xs,ys,H,STABILITY):
   u1=u
   x1=x-xs # shift the coordinates so that stack is centre point
   y1=y-ys

   # components of u (wind) in x and y directions
   wx=u1*np.sin((dir1-180.)*np.pi/180.)
   wy=u1*np.cos((dir1-180.)*np.pi/180.)

   # Angle between point x, y and the wind direction, so use scalar product:
   dot_product=wx*x1+wy*y1
   magnitudes=u1*np.sqrt(x1**2.+y1**2.) 
   subtended=np.arccos(dot_product/(magnitudes+1e-15))
   
   # distance to point x,y from stack
   hypotenuse=np.sqrt(x1**2.+y1**2.)

   # distance along the wind direction to perpendilcular line that intesects
   # x,y
   downwind=np.cos(subtended)*hypotenuse

   # Now calculate distance cross wind.
   crosswind=np.sin(subtended)*hypotenuse

   ind=np.where(downwind>0.)
   C=np.zeros((len(x),len(y)))
   
   # calculate sigmas based on stability and distance downwind
   (sig_y,sig_z)=calc_sigmas(STABILITY,downwind)
       

   C[ind]=Q/(2.*np.pi*u1*sig_y[ind]*sig_z[ind]) \
       * np.exp(-crosswind[ind]**2./(2.*sig_y[ind]**2.))  \
       *(np.exp(-(z[ind]-H)**2./(2.*sig_z[ind]**2.)) + \
       np.exp(-(z[ind]+H)**2./(2.*sig_z[ind]**2.)) )
   return C
