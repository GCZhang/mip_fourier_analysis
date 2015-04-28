# Python code
# Author: Bruno Turcksin
# Date: 2011-11-13 14:47:10.809852

#----------------------------------------------------------------------------#
## Class FINITE_ELEMENT                                                     ##
#----------------------------------------------------------------------------#

"""Base class for the finite elements"""

import numpy as np

class FINITE_ELEMENT:
  """Base class for the finite elements."""

  def __init__(self,x,y,fe_id) :

    self.x = x
    self.y = y
    x_1 = x[2]-x[0]
    x_2 = x[3]-x[1]
    y_1 = y[2]-y[0]
    y_2 = y[3]-y[1]
    self.area = 0.5*np.abs(x_1*y_2-x_2*y_1)
    self.fe_id = fe_id

#----------------------------------------------------------------------------#

  def Build_upwind_matrices(self) :
    """Compute all upwind and downwind matrices for a cell."""

    self.bottom_down = np.zeros((4,4))
    self.bottom_down[0,0] = self.bottom_edge_mass_matrix[0,0]
    self.bottom_down[0,1] = self.bottom_edge_mass_matrix[0,1]
    self.bottom_down[1,0] = self.bottom_edge_mass_matrix[1,0]
    self.bottom_down[1,1] = self.bottom_edge_mass_matrix[1,1]

    self.bottom_up = np.zeros((4,4))
    self.bottom_up[0,2] = self.bottom_coupling_edge_mass_matrix[0,0]
    self.bottom_up[0,3] = self.bottom_coupling_edge_mass_matrix[0,1]
    self.bottom_up[1,2] = self.bottom_coupling_edge_mass_matrix[1,0]
    self.bottom_up[1,3] = self.bottom_coupling_edge_mass_matrix[1,1]

    self.top_down = np.zeros((4,4))
    self.top_down[2,2] = self.top_edge_mass_matrix[0,0]
    self.top_down[2,3] = self.top_edge_mass_matrix[0,1]
    self.top_down[3,2] = self.top_edge_mass_matrix[1,0]
    self.top_down[3,3] = self.top_edge_mass_matrix[1,1]

    self.top_up = np.zeros((4,4))
    self.top_up[2,0] = self.top_coupling_edge_mass_matrix[0,0]
    self.top_up[2,1] = self.top_coupling_edge_mass_matrix[0,1]
    self.top_up[3,0] = self.top_coupling_edge_mass_matrix[1,0]
    self.top_up[3,1] = self.top_coupling_edge_mass_matrix[1,1]

    self.left_down = np.zeros((4,4))
    self.left_down[0,0] = self.left_edge_mass_matrix[0,0]
    self.left_down[0,3] = self.left_edge_mass_matrix[0,1]
    self.left_down[3,0] = self.left_edge_mass_matrix[1,0]
    self.left_down[3,3] = self.left_edge_mass_matrix[1,1]

    self.left_up = np.zeros((4,4))
    self.left_up[0,1] = self.left_coupling_edge_mass_matrix[0,0]
    self.left_up[0,2] = self.left_coupling_edge_mass_matrix[0,1]
    self.left_up[3,1] = self.left_coupling_edge_mass_matrix[1,0]
    self.left_up[3,2] = self.left_coupling_edge_mass_matrix[1,1]

    self.right_down = np.zeros((4,4))
    self.right_down[1,1] = self.right_edge_mass_matrix[0,0]
    self.right_down[1,2] = self.right_edge_mass_matrix[0,1]
    self.right_down[2,1] = self.right_edge_mass_matrix[1,0]
    self.right_down[2,2] = self.right_edge_mass_matrix[1,1]

    self.right_up = np.zeros((4,4))
    self.right_up[1,0] = self.right_coupling_edge_mass_matrix[0,0]
    self.right_up[1,3] = self.right_coupling_edge_mass_matrix[0,1]
    self.right_up[2,0] = self.right_coupling_edge_mass_matrix[1,0]
    self.right_up[2,3] = self.right_coupling_edge_mass_matrix[1,1]
