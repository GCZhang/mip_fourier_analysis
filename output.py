# Python code
# Author: Bruno Turcksin
# Date: 2011-11-14 17:50:30.896960

#----------------------------------------------------------------------------#
## Module output                                                            ##
#----------------------------------------------------------------------------#

"""Module to be used in ipython to look at the spectral radius and the
condition number."""

import numpy as np
import pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def Create_figure(filename, condition_number = False):
    """Create a figure in mplot3d."""

# Load the mesh, the spectral radius and the condition number
    data = np.load(filename+'.npz')
    lambda_x = data['lambda_x']
    lambda_y = data['lambda_y']
    rho = data['rho']

    x, y = np.meshgrid(lambda_x, lambda_y)
    z = np.array(rho)

    fig_1 = pylab.figure(1)
    ax_1 = fig_1.add_subplot(111,projection='3d')
    surf_1 = ax_1.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax_1.set_zlim(0.99*z.min(),1.01*z.max())
    pylab.title('Largest eigenvalues')
    pylab.xlabel("lambda_x")
    pylab.ylabel("lambda_y")
    fig_1.colorbar(surf_1, shrink=0.5)
    fig_1.show()
    
    if condition_number == True:
        fig_2 = pylab.figure(2)
        ax_2 = fig_2.add_subplot(111,projection='3d')
        kappa = data['kappa']
        surf_2 = ax_2.plot_surface(x, y, kappa, rstride=1, cstride=1, cmap=cm.coolwarm)
        pylab.title('Condition number')
        pylab.xlabel("lambda_x")
        pylab.ylabel("lambda_y")
        fig_2.colorbar(surf_1, shrink=0.5)
        fig_2.show()
