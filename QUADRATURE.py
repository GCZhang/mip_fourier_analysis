# Python code
# Author: Bruno Turcksin
# Date: 2011-11-11 17:51:39.155307

#----------------------------------------------------------------------------#
## Class QUADRATURE                                                         ##
#----------------------------------------------------------------------------#

"""Build the quadrature"""

import numpy as np
import scipy.linalg
import scipy.misc as sci

class QUADRATURE:
  """Build the quadrature and the Galerkin version of the quadrature. Create 
  the M and D matrices."""

  def __init__(self,sn,L_max,galerkin) :

    self.sn = sn
    self.n_dir = int(self.sn*(self.sn+2)/2)
    self.galerkin = galerkin
    self.L_max = L_max
    if (self.galerkin==True) :
      self.n_mom = self.n_dir
      if (sn!=L_max) :
        raise AssertionError('sn!=L_max')
    else :
      self.n_mom = int((self.L_max+1)*(self.L_max+2)/2)
    self.Build_quadrature()

#----------------------------------------------------------------------------#

  def Build_quadrature(self) :
    """Build the quadrature, i.e. M, D and omega (direction vector)."""

# Compute omega on one quadrant
    self.Build_quadrant()

# Compute omega by deploying the quadrant 
    self.Deploy_quadrant()

# Compute the spherical harmonics
    self.Compute_harmonics()

# Compute D
    if self.galerkin == True :
      self.D = scipy.linalg.inv(self.M)
    else :
      self.D = np.dot(self.M.transpose(),np.diag(self.weight))

#----------------------------------------------------------------------------#

  def Build_quadrant(self) :
    """Build omega and weight for one quadrant. This function is purely
    virtual"""

    pass

#----------------------------------------------------------------------------#

  def Deploy_quadrant(self) :
    """Compute omega and the weights by deploing the quadrants."""

    n_dir_oct = int(self.n_dir/4)
    offset = 0
    for i_octant in range(0, 4) :
      if i_octant != 0 :
        for i in range (0, n_dir_oct) :
# Copy omega and weight 
          self.weight[i+offset] = self.weight[i]
          self.omega[i+offset,2] = self.omega[i,2]
# Correct omega signs
          if i_octant == 1 :
            self.omega[i+offset,0] = self.omega[i,0]
            self.omega[i+offset,1] = -self.omega[i,1]
          elif i_octant == 2 :
            self.omega[i+offset,0] = -self.omega[i,0]
            self.omega[i+offset,1] = -self.omega[i,1]
          else :
            self.omega[i+offset,0] = -self.omega[i,0]
            self.omega[i+offset,1] = self.omega[i,1]
      offset += n_dir_oct

    sum_weight = 0.
    for i in range(0, n_dir_oct) :
      sum_weight += 4 * self.weight[i]
    self.weight[:] = self.weight[:]/sum_weight

#----------------------------------------------------------------------------#

  def Compute_harmonics(self) :
    """Compute the spherical harmonics and build the matrix M."""

    Ye = np.zeros((self.L_max+1,self.L_max+1,self.n_dir))
    Yo = np.zeros((self.L_max+1,self.L_max+1,self.n_dir))

    phi = np.zeros((self.n_dir,1))
    for i in range(0, self.n_dir) :
      phi[i] = np.arctan(self.omega[i,1]/self.omega[i,0])
      if self.omega[i,0] < 0. :
        phi[i] = phi[i] + np.pi

    for l in range(0, self.L_max+1) :
      for m in range(0, l+1) :
        P_ml =  scipy.special.lpmv(m,l,self.omega[:,2])
# Normalization of the associated Legendre polynomials
        if m == 0 :
          norm_P = P_ml
        else :
          norm_P = (-1.0)**m*np.sqrt(2*sci.factorial(l-m)/sci.factorial(l+m))\
              *P_ml
        size = norm_P.shape
        for i in range(0, size[0]) :
          Ye[l,m,i] = norm_P[i]*np.cos(m*phi[i])
          Yo[l,m,i] = norm_P[i]*np.sin(m*phi[i])

# Build the matrix M 
    self.sphr = np.zeros((self.n_dir,self.n_mom))
    self.M = np.zeros((self.n_dir,self.n_mom))
    if self.galerkin == True :
      for i in range(0, self.n_dir) :
        pos = 0
        for l in range(0, self.L_max+1) :
          fact = 2*l+1
          for m in range(l, -1, -1) :
# do not use the EVEN when m+l is odd for L<sn of L=sn and m=0
            if l<self.sn and np.fmod(m+l,2)==0 :
              self.sphr[i,pos] = Ye[l,m,i]
              self.M[i,pos] = fact*self.sphr[i,pos]
              pos += 1
          for m in range(1, l+1) :
# do not use the ODD when m+l is odd for l<=sn
            if l<=self.sn and  np.fmod(m+l,2)==0 :
              self.sphr[i,pos] = Yo[l,m,i]
              self.M[i,pos] = fact*self.sphr[i,pos]
              pos += 1
    else :
      for i in range(0, self.n_dir) :
        pos = 0
        for l in range(0, self.L_max+1) :
          fact = 2*l+1
          for m in range(l, -1, -1) :
# do not use the EVEN when m+l is odd 
            if np.fmod(m+l,2)==0 :
              self.sphr[i,pos] = Ye[l,m,i]
              self.M[i,pos] = fact*self.sphr[i,pos]
              pos += 1
          for m in range(1, l+1) :
# do not use the ODD when m+l is odd 
            if np.fmod(m+l,2)==0 :
              self.sphr[i,pos] = Yo[l,m,i]
              self.M[i,pos] = fact*self.sphr[i,pos]
              pos += 1
