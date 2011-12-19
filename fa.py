#! /usr/bin/env python
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
import utils

grid_x = np.array([0.,0.25,0.5,0.75,1.,0.,0.25,0.5,0.75,1.,0.,0.25,0.5,0.75,1.,
  0.,0.25,0.5,0.75,1.,0.,0.25,0.5,0.75,1.])
grid_y = np.array([0.,0.,0.,0.,0.,0.25,0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5,0.5,
  0.75,0.75,0.75,0.75,0.75,1.,1.,1.,1.,1.])
nx_cells = 4
ny_cells = 4
grid_x = np.array([0.,1.,0.,1.])
grid_y = np.array([0.,0.,1.,1.])
nx_cells = 1
ny_cells = 1
N = 5
solver_type = "SI"
condition_number = False
sn = 2
# CANNOT BE DIFFERENT THAT 0. Otherwise problem when D and M have to be
# multiplied
L_max = 0
galerkin = False
fe_type ="BLD"  
quad_type = "GLC"
#quad_type = "LS"
prec = False
filename = "transport"
# First element of cross section is the total cross section. The rest is the
# scattering cross section
cross_section = np.array([1,0.999999])

if grid_x.shape!=grid_y.shape :
  utils.abort("size of grid_x is not equal to size grid_y.")

if quad_type=="GLC" :
  quad = GLC.GLC(sn,L_max,galerkin)
elif quad_type=="LS" :
  quad = LS.LS(sn,L_max,galerkin)
else :
  utils.Abort("This quadrature does not exist.")
dof_handler = DOF_HANDLER.DOF_HANDLER(nx_cells,ny_cells,grid_x,grid_y,fe_type)
transport = TRANSPORT.TRANSPORT(dof_handler,quad,cross_section,solver_type,
    prec)

lambda_x = np.linspace(0,np.pi,N)
lambda_y = np.linspace(0,np.pi,N)
rho = np.zeros((N,N))

if condition_number==True :
  kappa = np.zeros((N,N))

for i in xrange(0,N) :
  for j in xrange(0,N) :
    lambdas = np.array([lambda_x[j],lambda_y[i]])
    print lambdas
    eig = transport.Compute_largest_eigenvalue(lambdas)
    print eig
    rho[i,j] = np.abs(eig.real)
    if condition_number==True :
      kappa[i,j] = transport.Compute_condition_number()

bounds = [(0.,2.*np.pi),(0.,2.*np.pi)]
initial_guess = np.array([0.,0.])
[x,fval,d] = scipy.optimize.fmin_l_bfgs_b(
    transport.Compute_largest_eigenvalue,initial_guess,args="m",approx_grad=1,
    bounds=bounds)
print x
print -fval
print d['warnflag'],d['funcalls'],d['task']

if condition_number==False :
  np.savez(filename,lambda_x=lambda_x,lambda_y=lambda_y,rho=rho)      
else :
  np.savez(filename,lambda_x=lambda_x,lambda_y=lambda_y,rho=rho,kappa=kappa) 
