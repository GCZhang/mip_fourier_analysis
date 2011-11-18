# Python code
# Author: Bruno Turcksin
# Date: 2011-11-13 14:47:10.809852

#----------------------------------------------------------------------------#
## Class FINITE_ELEMENT                                                     ##
#----------------------------------------------------------------------------#

"""Contain the finite elements"""

import numpy as np
import utils

class FINITE_ELEMENT(object) :
  """Contain the different finite elements : BLD and PWLD (2D) :
    3--2
    |  |
    0--1"""

  def __init__(self,x,y,fe_id,fe_type) :

    super(FINITE_ELEMENT,self).__init__()
    self.x = x
    self.y = y
    self.fe_id = fe_id
    if (fe_type=="BLD") :
      self.delta_x = self.x[1]-self.x[0]
      self.delta_y = self.y[3]-self.y[0]
      self.Build_bld()
    elif (fe_type=="PWLD") :
      self.Build_pwld()
    else :
      utils.abort("Unkwnon discretization")
    self.Build_upwind_matrices()

#----------------------------------------------------------------------------#

  def Build_bld(self) :
    """Build the BiLD finite elements. Basis functions : b_0 = (1-x)(1-y),
    b_1 = x(1-y), b_2 = xy and b_3 = (1-x)y"""

    self.Build_bld_2d()
    self.Build_bld_1d()

#----------------------------------------------------------------------------#

  def Build_bld_2d(self) :
    """Build the mass matrix, two gradient matrices and the stiffness matrix
    of BiLD."""

    self.mass_matrix = self.delta_x*self.delta_y/36.*np.array([[4.,2.,1.,2.],
        [2.,4.,2.,1.],[1.,2.,4.,2.],[2.,1.,2.,4.]])
    self.x_grad_matrix = self.delta_y/12.*np.array([[-2.,-2.,-1.,-1.],
      [2.,2.,1.,1.],[1.,1.,2.,2.],[-1.,-1.,-2.,-2.]])
    self.y_grad_matrix = self.delta_x/12.*np.array([[-2.,-1.,-1.,-2.],
      [-1.,-2.,-2.,-1.],[1.,2.,2.,1.],[2.,1.,1.,2.]])
    self.stiffness_matrix = self.delta_y/(6.*self.delta_x)*\
        np.array([[2.,-2.,-1.,1],[-2.,2.,1.,-1.],[-1.,1.,2.,-2.],\
        [1.,-1.,-2.,2.]]) + self.delta_x/(6.*self.delta_y)*\
        np.array([[2.,1.,-1.,2.],[1.,2.,-2.,-1.],[-1.,-2.,2.,1.],\
        [-2.,-1.,1.,2.]])

#----------------------------------------------------------------------------#

  def Build_bld_1d(self) :
    """Build the edge mass matrices, edge_deln_matrix and
    acroos_edge_deln_matrix."""

    h_ratio = self.delta_x/(6.*self.delta_y)
    v_ratio = self.delta_y/(6.*self.delta_x)
    
    self.horizontal_edge_mass_matrix = self.delta_x/6.*np.array([[2.,1.],
      [1.,2.]])
    self.horizontal_coupling_edge_mass_matrix = self.delta_x/6.*np.array(
        [[1.,2.],[2.,1.]])
    self.vertical_edge_mass_matrix = self.delta_y/6.*np.array([[2.,1.],
      [1.,2.]])
    self.vertical_coupling_edge_mass_matrix = self.delta_y/6.*np.array(
        [[2.,1.],[1.,2.]])
    
    left = v_ratio*np.array([[-2.,0.,0.,-1.],[2.,0.,0.,1.],[1.,0.,0.,2.],
      [-1.,0.,0.,-2.]])
    right = v_ratio*np.array([[0.,-2.,-1.,0.],[0.,2.,1.,0.],[0.,1.,2.,0.],
      [0.,-1.,-2.,0.]])
    bottom = h_ratio*np.array([[-2.,-1.,0.,0.],[-1.,-2.,0.,0.],[1.,2.,0.,0.],
      [2.,1.,0.,0.]])
    top = h_ratio*np.array([[0.,0.,-1.,-2.],[0.,0.,-2.,-1.],[0.,0.,2.,1.],
      [0.,0.,1.,2.]])

    self.edge_deln_matrix = {'left' : left, 'right' : right, 'bottom' :
        bottom, 'top' : top}

    left = h_ratio*np.array([[0.,-2.,-1.,0.],[0.,2.,1.,0.],[0.,1.,2.,0.],
      [0.,-1.,-2.,0.]])
    right = h_ratio*np.array([[-2.,0.,0.,-1.],[2.,0.,0.,1.],[1.,0.,0.,2.],
      [-1.,0.,0.,-2]])
    bottom = v_ratio*np.array([[0.,0.,-1.,-2.],[0.,0.,-2.,-1.],[0.,0.,2.,1.],
      [0.,0.,1.,2.]])
    top = v_ratio*np.array([[-2.,-1.,0.,0.],[-1.,-2.,0.,0.],[1.,2.,0.,0.],
      [2.,1.,0.,0.]])
    self.across_edge_deln_matrix = {'left' : left, 'right' : right, 'bottom' :
        bottom, 'top' : top}

#----------------------------------------------------------------------------#

  def Build_upwind_matrices(self) :
    """Compute all upwind and downwind matrices for a cell."""

    self.bottom_down = np.zeros((4,4))
    self.bottom_down[0,0] = self.horizontal_edge_mass_matrix[0,0]
    self.bottom_down[0,1] = self.horizontal_edge_mass_matrix[0,1]
    self.bottom_down[1,0] = self.horizontal_edge_mass_matrix[1,0]
    self.bottom_down[1,1] = self.horizontal_edge_mass_matrix[1,1]

    self.bottom_up = np.zeros((4,4))
    self.bottom_up[0,2] = self.horizontal_coupling_edge_mass_matrix[0,0]
    self.bottom_up[0,3] = self.horizontal_coupling_edge_mass_matrix[0,1]
    self.bottom_up[1,2] = self.horizontal_coupling_edge_mass_matrix[1,0]
    self.bottom_up[1,3] = self.horizontal_coupling_edge_mass_matrix[1,1]

    self.top_down = np.zeros((4,4))
    self.top_down[2,2] = self.horizontal_edge_mass_matrix[0,0]
    self.top_down[2,3] = self.horizontal_edge_mass_matrix[0,1]
    self.top_down[3,2] = self.horizontal_edge_mass_matrix[1,0]
    self.top_down[3,3] = self.horizontal_edge_mass_matrix[1,1]

    self.top_up = np.zeros((4,4))
    self.top_up[2,0] = self.horizontal_coupling_edge_mass_matrix[0,0]
    self.top_up[2,1] = self.horizontal_coupling_edge_mass_matrix[0,1]
    self.top_up[3,0] = self.horizontal_coupling_edge_mass_matrix[1,0]
    self.top_up[3,1] = self.horizontal_coupling_edge_mass_matrix[1,1]

    self.left_down = np.zeros((4,4))
    self.left_down[0,0] = self.vertical_edge_mass_matrix[0,0]
    self.left_down[0,3] = self.vertical_edge_mass_matrix[0,1]
    self.left_down[3,0] = self.vertical_edge_mass_matrix[1,0]
    self.left_down[3,3] = self.vertical_edge_mass_matrix[1,1]

    self.left_up = np.zeros((4,4))
    self.left_up[0,1] = self.vertical_coupling_edge_mass_matrix[0,0]
    self.left_up[0,2] = self.vertical_coupling_edge_mass_matrix[0,1]
    self.left_up[3,1] = self.vertical_coupling_edge_mass_matrix[1,0]
    self.left_up[3,2] = self.vertical_coupling_edge_mass_matrix[1,1]

    self.right_down = np.zeros((4,4))
    self.right_down[1,1] = self.vertical_edge_mass_matrix[0,0]
    self.right_down[1,2] = self.vertical_edge_mass_matrix[0,1]
    self.right_down[2,1] = self.vertical_edge_mass_matrix[1,0]
    self.right_down[2,2] = self.vertical_edge_mass_matrix[1,1]

    self.right_up = np.zeros((4,4))
    self.right_up[1,0] = self.vertical_coupling_edge_mass_matrix[0,0]
    self.right_up[1,3] = self.vertical_coupling_edge_mass_matrix[0,1]
    self.right_up[2,0] = self.vertical_coupling_edge_mass_matrix[1,0]
    self.right_up[2,3] = self.vertical_coupling_edge_mass_matrix[1,1]
