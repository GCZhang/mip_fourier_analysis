# Python code
# Author: Bruno Turcksin
# Date: 2011-11-21 12:26:43.443470

#----------------------------------------------------------------------------#
## Class BLD                                                                ##
#----------------------------------------------------------------------------#

"""Contain the BLD finite elements."""

import numpy as np
import utils
import FINITE_ELEMENT as FE

class BLD(FE.FINITE_ELEMENT) :
  """Contain the BLD finite elements :
    3--2
    |  |
    0--1
    Basis functions : b_0=(1-x)(1-y), b_1=x(1-y), b_2=xy and b_3=(1-x)y """

  def __init__(self,x,y,fe_id) :

    super(BLD,self).__init__(x,y,fe_id)
    self.delta_x = self.x[1]-self.x[0]
    self.delta_y = self.y[3]-self.y[0]
    self.Build_bld_1d()
    self.Build_bld_2d()
    self.Build_upwind_matrices()

#----------------------------------------------------------------------------#

  def Build_bld_1d(self) :
    """Build the edge mass matrices, edge_deln_matrix and
    across_edge_deln_matrix."""

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

    self.bottom_edge_mass_matrix = self.horizontal_edge_mass_matrix
    self.top_edge_mass_matrix = self.horizontal_edge_mass_matrix
    self.left_edge_mass_matrix = self.vertical_edge_mass_matrix
    self.right_edge_mass_matrix = self.vertical_edge_mass_matrix

    self.bottom_coupling_edge_mass_matrix =\
        self.horizontal_coupling_edge_mass_matrix
    self.top_coupling_edge_mass_matrix =\
        self.horizontal_coupling_edge_mass_matrix
    self.left_coupling_edge_mass_matrix =\
        self.vertical_coupling_edge_mass_matrix
    self.right_coupling_edge_mass_matrix =\
        self.vertical_coupling_edge_mass_matrix
    
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

  def Build_bld_2d(self) :
    """Build the mass matrix, two gradient matrices and the stiffness matrix
    of BLD."""

    self.mass_matrix = self.delta_x*self.delta_y/36.*np.array([[4.,2.,1.,2.],
        [2.,4.,2.,1.],[1.,2.,4.,2.],[2.,1.,2.,4.]])
    self.x_grad_matrix = self.delta_y/12.*np.array([[-2.,-2.,-1.,-1.],
      [2.,2.,1.,1.],[1.,1.,2.,2.],[-1.,-1.,-2.,-2.]])
    self.y_grad_matrix = self.delta_x/12.*np.array([[-2.,-1.,-1.,-2.],
      [-1.,-2.,-2.,-1.],[1.,2.,2.,1.],[2.,1.,1.,2.]])
    self.stiffness_matrix = self.delta_y/(6.*self.delta_x)*\
        np.array([[2.,-2.,-1.,1],[-2.,2.,1.,-1.],[-1.,1.,2.,-2.],\
        [1.,-1.,-2.,2.]]) + self.delta_x/(6.*self.delta_y)*\
        np.array([[2.,1.,-1.,-2.],[1.,2.,-2.,-1.],[-1.,-2.,2.,1.],\
        [-2.,-1.,1.,2.]])
