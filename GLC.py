# Python code
# Author: Bruno Turcksin
# Date: 2011-12-13 15:38:33.563356

#----------------------------------------------------------------------------#
## Class GLC                                                                ##
#----------------------------------------------------------------------------#

"""Contain the Gauss-Legendre-Chebyshev quadrature"""

import numpy as np
import scipy.special.orthogonal
import QUADRATURE as QUAD

class GLC(QUAD.QUADRATURE):
  """Build the Gauss-Legendre-Chebyshev quadrature of arbitrary order."""

  def __init__(self,sn,L_max,galerkin) :

    QUAD.QUADRATURE.__init__(self,sn,L_max,galerkin)

#----------------------------------------------------------------------------#

  def Build_quadrant(self) :
    """Build omega and weight for one quadrant."""

# Compute the Gauss-Legendre quadrature
    [self.polar_nodes,self.polar_weight] = scipy.special.orthogonal.p_roots(self.sn) 

# Compute the Chebyshev quadrature
    [self.azith_nodes,self.azith_weight] = self.Chebyshev()

    self.cos_theta = np.zeros((self.sn/2,1))
    for i in range(0, int(self.sn/2)) :
      self.cos_theta[i] = np.real(self.polar_nodes[self.sn/2+i])
    self.sin_theta = np.sqrt(1-self.cos_theta**2)
    
    self.omega = np.zeros((self.n_dir,3))
    self.weight = np.zeros((self.n_dir))

    pos = 0
    offset = 0
    for i in range(0, int(self.sn/2)) :
      for j in range(0, int(self.sn/2-i)) :
        self.omega[pos,0] = self.sin_theta[i]*np.cos(self.azith_nodes[j+offset])
        self.omega[pos,1] = self.sin_theta[i]*np.sin(self.azith_nodes[j+offset])
        self.omega[pos,2] = self.cos_theta[i]
        self.weight[pos] = self.polar_weight[self.sn/2+i]*\
            self.azith_weight[j+offset]
        pos += 1
      offset += self.sn/2-i  

#----------------------------------------------------------------------------#

  def Chebyshev(self) :
    """Build the Chebyshev quadrature in a quadrant."""

    size = 0
    for i in range(1, int(self.sn/2+1)) :
      size += i
    nodes = np.zeros((size,1))
    weight = np.zeros((size))

    pos = 0
    for i in range(0, int(self.sn/2)) :
      for j in range(0, int(self.sn/2-i)) :
        nodes[pos] = (np.pi/2.)/(self.sn/2-i)*j+(np.pi/4.)/(self.sn/2-i)
        weight[pos] = np.pi/(2.*(self.sn/2-i))
        pos += 1

    return nodes,weight
