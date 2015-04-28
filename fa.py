#! /usr/bin/env python3
#----------------------------------------------------------------------------#
# Python code
# Author: Bruno Turcksin
# Date: 2011-11-11 17:48:03.487349
#----------------------------------------------------------------------------#

import numpy as np
import scipy.optimize
import GLC
import LS
import DOF_HANDLER
import TRANSPORT

grid_x = np.array([0.,0.5,1.,0.,0.5,1.,0.,0.5,1.])
grid_y = np.array([0.,0.,0.,0.5,0.5,0.5,1.,1.,1.])
nx_cells = 2
ny_cells = 2
grid_x = np.array([0.,1.,0.,1.])
grid_y = np.array([0.,0.,1.,1.])
nx_cells = 1
ny_cells = 1
N = 10
solver_type = 'SI'
condition_number = False
sn = 16
L_max = 0
galerkin = False
fe_type = 'BLD'
quad_type = 'GLC'
#quad_type = 'LS'
prec = True
filename = 'transport'
# First element of cross section is the total cross section. The rest is the
# scattering cross section
cross_section = np.array([[[0.01, 0.00999999]]])

if grid_x.shape != grid_y.shape:
    raise AssertionError('Size of grid_x is not equal to size grid_y.')

if quad_type == 'GLC':
    quad = GLC.GLC(sn,L_max,galerkin)
elif quad_type == 'LS':
    quad = LS.LS(sn,L_max,galerkin)
else:
    raise NotImplementedError('This quadrature is not implented')
dof_handler = DOF_HANDLER.DOF_HANDLER(nx_cells, ny_cells, grid_x, grid_y,
        fe_type, cross_section, quad.n_mom)
transport = TRANSPORT.TRANSPORT(dof_handler,quad,solver_type,
    prec)

lambda_x = np.linspace(0,np.pi,N)
lambda_y = np.linspace(0,np.pi,N)
rho = np.zeros((N,N))

if condition_number == True:
    kappa = np.zeros((N,N))

for i in range(0,N):
    for j in range(0,N):
        lambdas = np.array([lambda_x[j],lambda_y[i]])
        eig = transport.Compute_largest_eigenvalue(lambdas)
        rho[i,j] = np.abs(eig.real)
        if condition_number == True:
            kappa[i,j] = transport.Compute_condition_number()

bounds = [(0.,2.*np.pi), (0.,2.*np.pi)]
fval = 0.
for i in range(0, 5):
    rand = np.random.rand(2,1)
    initial_guess = np.array([rand[0,0],rand[1,0]])
    [x2,fval2,d] = scipy.optimize.fmin_l_bfgs_b(
            transport.Compute_largest_eigenvalue,initial_guess,args="m",
            approx_grad=1,bounds=bounds)
    print('Temporary frequencies', x2) 
    print('Temporary largest eigenvalue', -fval2) 
    if fval2 < fval:
        x = x2.copy()
        fval = fval2
    print('Warning flag', d['warnflag'], 'Number of function calls', d['funcalls'])
print('Frequencies', x)
print('Largest eigenvalue', -fval)

if condition_number == False:
    np.savez(filename, lambda_x=lambda_x, lambda_y=lambda_y, rho=rho)      
else:
    np.savez(filename, lambda_x=lambda_x, lambda_y=lambda_y, rho=rho, kappa=kappa) 
