# Python code
# Author: Bruno Turcksin
# Date: 2011-11-14 17:50:30.896960

#----------------------------------------------------------------------------#
## Module output                                                            ##
#----------------------------------------------------------------------------#

"""Module to be used in ipython to look at the spectral radius and the
condition number."""

import numpy as np
from mayavi import mlab

def Create_figure(filename,condition_number=False) :
  """Open mayavi."""

# Load the mesh, the spectral radius and the condition number
  data = np.load(filename+'.npz')
  lambda_x = data['lambda_x']
  lambda_y = data['lambda_y']
  rho = data['rho']

  mlab.surf(lambda_x,lambda_y,rho,name='Largest eigenvalues')
  mlab.colorbar(orientation='vertical')
  mlab.xlabel("lambda_x")
  mlab.ylabel("lambda_y")
  mlab.view(0,0,distance='auto',focalpoint='auto')
  
  if condition_number==True :
    kappa = data['kappa']
    mlab.surf(lambda_x,lambda_y,kappa,kappa,name='Condition number')
