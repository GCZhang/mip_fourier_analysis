# Python code
# Author: Bruno Turcksin
# Date: 2011-11-21 12:44:44.099599

#----------------------------------------------------------------------------#
## Class PWLD                                                               ##
#----------------------------------------------------------------------------#

"""Contain the PWLD finite elements."""

import numpy as np
import utils
import FINITE_ELEMENT as FE

class PWLD(FE.FINITE_ELEMENT) :
  """Contain the PWLD finite elements for quadrangles"""

  def __init__(self,x,y,fe_id) :

    super(PWLD,self).__init__(x,y,fe_id)
    self.x_c = 0.25*(self.x[0]+self.x[1]+self.x[2]+self.x[3])
    self.y_c = 0.25*(self.y[0]+self.y[1]+self.y[2]+self.y[3])
    self.Build_pwld_1d()
    self.Build_pwld_2d()
    self.Build_upwind_matrices()

#----------------------------------------------------------------------------#

  def Build_pwld_1d(self) :
    """Build the edge mass matrices, edge_deln_matrix and
    across_edge_deln_matrix."""

    L_b = np.sqrt((self.x[1]-self.x[0])**2+(self.y[1]-self.y[0])**2)
    L_r = np.sqrt((self.x[2]-self.x[1])**2+(self.y[2]-self.y[1])**2)
    L_t = np.sqrt((self.x[3]-self.x[2])**2+(self.y[3]-self.y[2])**2)
    L_l = np.sqrt((self.x[3]-self.x[0])**2+(self.y[3]-self.y[0])**2)

    self.bottom_edge_mass_matrix = L_b/6.*np.array([[2.,1.],[1.,2.]])
    self.bottom_coupling_edge_mass_matrix = L_b/6.*np.array([[1.,2.],[2.,1.]])
    self.top_edge_mass_matrix = L_t/6.*np.array([[2.,1.],[1.,2.]])
    self.top_coupling_edge_mass_matrix = L_t/6.*np.array([[1.,2.],[2.,1.]])
    self.left_edge_mass_matrix = L_l/6.*np.array([[2.,1.],[1.,2.]])
    self.left_coupling_edge_mass_matrix = L_l/6.*np.array([[2.,1.],[1.,2.]])
    self.right_edge_mass_matrix = L_r/6.*np.array([[2.,1.],[1.,2.]])
    self.right_coupling_edge_mass_matrix = L_r/6.*np.array([[2.,1.],[1.,2.]])

#----------------------------------------------------------------------------#

  def Build_pwld_2d(self) :
    """Build the mass matrix, two gradient matrices and the stiffness matrix
    of PWLD."""

    self.mass_matrix = np.zeros((4,4))
    self.x_grad_matrix = np.zeros((4,4))
    self.y_grad_matrix = np.zeros((4,4))
    self.stiffness_matrix = np.zeros((4,4))

# Build the matrices by looping over the ``sides''
    mass_side = np.array([[2.,1.,1.],[1.,2.,1.],[1.,1.,2.]])
    for side in xrange(0,4) :
      a = side
      b = (side+1)%4
      x0 = self.x[a]
      x1 = self.x[b]
      x2 = self.x_c
      y0 = self.y[a]
      y1 = self.y[b]
      y2 = self.y_c

      x2_x1 = x2-x1
      x1_x2 = x1-x2
      x0_x2 = x0-x2
      x2_x0 = x2-x0
      x1_x0 = x1-x0
      x0_x1 = x0-x1
      y1_y2 = y1-y2
      y2_y1 = y2-y1
      y2_y0 = y2-y0
      y0_y2 = y0-y2
      y0_y1 = y0-y1
      y1_y0 = y1-y0

      a_00 = 0.5*(y2_y1**2+x2_x1**2)
      a_01 = -0.5*(y2_y0*y2_y1+x2_x0*x2_x1)
      a_02 = -0.5*(y0_y1*y2_y1+x0_x1*x2_x1)
      a_11 = 0.5*(y2_y0**2+x2_x0**2)
      a_12 = 0.5*(y2_y0*y0_y1+x0_x2*x1_x0)
      a_22 = 0.5*(y0_y1**2+x1_x0**2)

      jacobian = np.abs(x1_x0*y2_y0-y1_y0*x2_x0)
      area = 0.5*jacobian

      mass_matrix = area/12*mass_side
      x_grad_matrix = 1./6.*np.array([[y1_y2,y1_y2,y1_y2],[y2_y0,y2_y0,y2_y0],
        [y0_y1,y0_y1,y0_y1]])
      y_grad_matrix = 1./6.*np.array([[x2_x1,x2_x1,x2_x1],[x0_x2,x0_x2,x0_x2],
        [x1_x0,x1_x0,x1_x0]])
      stiffness_matrix = 1./jacobian*np.array([[a_00,a_01,a_02],
        [a_01,a_11,a_12],[a_02,a_12,a_22]])

      self.mass_matrix[a,a] += mass_matrix[0,0]
      self.mass_matrix[a,b] += mass_matrix[0,1]
      self.mass_matrix[b,a] += mass_matrix[1,0]
      self.mass_matrix[b,b] += mass_matrix[1,1]

      self.x_grad_matrix[a,a] += x_grad_matrix[0,0]
      self.x_grad_matrix[a,b] += x_grad_matrix[0,1]
      self.x_grad_matrix[b,a] += x_grad_matrix[1,0]
      self.x_grad_matrix[b,b] += x_grad_matrix[1,1]

      self.y_grad_matrix[a,a] += y_grad_matrix[0,0]
      self.y_grad_matrix[a,b] += y_grad_matrix[0,1]
      self.y_grad_matrix[b,a] += y_grad_matrix[1,0]
      self.y_grad_matrix[b,b] += y_grad_matrix[1,1]

      self.stiffness_matrix[a,a] += stiffness_matrix[0,0]
      self.stiffness_matrix[a,b] += stiffness_matrix[0,1]
      self.stiffness_matrix[b,a] += stiffness_matrix[1,0]
      self.stiffness_matrix[b,b] += stiffness_matrix[1,1]
      
      for i in xrange(0,4) :
        self.mass_matrix[a,i] += 0.25*mass_matrix[0,2]
        self.mass_matrix[b,i] += 0.25*mass_matrix[1,2]
        self.mass_matrix[i,a] += 0.25*mass_matrix[2,0]
        self.mass_matrix[i,b] += 0.25*mass_matrix[2,1]

        self.x_grad_matrix[a,i] += 0.25*x_grad_matrix[0,2]
        self.x_grad_matrix[b,i] += 0.25*x_grad_matrix[1,2]
        self.x_grad_matrix[i,a] += 0.25*x_grad_matrix[2,0]
        self.x_grad_matrix[i,b] += 0.25*x_grad_matrix[2,1]

        self.y_grad_matrix[a,i] += 0.25*y_grad_matrix[0,2]
        self.y_grad_matrix[b,i] += 0.25*y_grad_matrix[1,2]
        self.y_grad_matrix[i,a] += 0.25*y_grad_matrix[2,0]
        self.y_grad_matrix[i,b] += 0.25*y_grad_matrix[2,1]

        self.stiffness_matrix[a,i] += 0.25*self.stiffness_matrix[0,2]
        self.stiffness_matrix[b,i] += 0.25*self.stiffness_matrix[1,2]
        self.stiffness_matrix[i,a] += 0.25*self.stiffness_matrix[2,0]
        self.stiffness_matrix[i,b] += 0.25*self.stiffness_matrix[2,1]

        for j in xrange(0,4) :
          self.mass_matrix[i,j] += 0.25**2*mass_matrix[2,2]
          self.x_grad_matrix[i,j] += 0.25**2*x_grad_matrix[2,2]
          self.y_grad_matrix[i,j] += 0.25**2*y_grad_matrix[2,2]
          self.stiffness_matrix[i,j] += 0.25**2*self.stiffness_matrix[2,2]
