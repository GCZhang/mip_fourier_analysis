# Python code
# Author: Bruno Turcksin
# Date: 2011-11-13 17:21:58.422971

#----------------------------------------------------------------------------#
## Class DOF_HANDLER                                                        ##
#----------------------------------------------------------------------------#

"""Contain the degrees of freedom handler"""

import numpy as np
import utils
import BLD
import PWLD

class DOF_HANDLER(object) :
  """Build the cells"""

  def __init__(self,nx_cells,ny_cells,x,y,fe_type) :

    super(DOF_HANDLER,self).__init__()
    self.nx_cells = nx_cells
    self.ny_cells = ny_cells
    self.n_cells = nx_cells*ny_cells
    self.x = x
    self.y = y
    self.fe_type = fe_type
    self.bottom = self.y[0]
    self.top = self.y[-1]
    self.left = self.x[0]
    self.right = self.x[self.nx_cells]
    self.Build_fe()

#----------------------------------------------------------------------------#

  def Build_fe(self) :
    """Build the finite elements."""

    x = np.zeros((4))
    y = np.zeros((4))
    self.grid = []

    for i in range(0, self.ny_cells) :
      for j in range(0, self.nx_cells) :
        x[0] = self.x[j+i*(self.nx_cells+1)]
        y[0] = self.y[j+i*(self.nx_cells+1)]
        x[1] = self.x[j+1+i*(self.nx_cells+1)]
        y[1] = self.y[j+1+i*(self.nx_cells+1)]
        x[2] = self.x[j+1+(i+1)*(self.nx_cells+1)]
        y[2] = self.y[j+1+(i+1)*(self.nx_cells+1)]
        x[3] = self.x[j+(i+1)*(self.nx_cells+1)]
        y[3] = self.y[j+(i+1)*(self.nx_cells+1)]

        if self.fe_type=="BLD" :
          self.grid.append(BLD.BLD(x.copy(),y.copy(),len(self.grid)))
        elif self.fe_type=="PWLD" :
          self.grid.append(PWLD.PWLD(x.copy(),y.copy(),len(self.grid)))
        else :
          utils.abort("Unknown discretization.")
