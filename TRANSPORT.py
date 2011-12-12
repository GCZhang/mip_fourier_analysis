# Python code
# Author: Bruno Turcksin
# Date: 2011-11-13 17:59:34.688980

#----------------------------------------------------------------------------#
## Class TRANSPORT                                                          ##
#----------------------------------------------------------------------------#

"""Build the transport matrix"""

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import utils
import MIP

class TRANSPORT(object) :
  """Build the transport matrix, i.e., everything the external source for SI
  and GMRES."""

  def __init__(self,dof_handler,quadrature,cross_section,solver_type,prec) :

    super(TRANSPORT,self).__init__()
    self.dof_handler = dof_handler
    self.quad = quadrature
    self.sigma_t = cross_section[0]
    self.sigma_s = np.zeros(self.quad.n_mom)
    self.sigma_s[0:(cross_section.shape[0]-1)] = cross_section[1:]
    self.solver_type = solver_type
    self.preconditioner = prec

#----------------------------------------------------------------------------#

  def Compute_largest_eigenvalue(self,lambda_x,lambda_y) :
    """Compute the largest (magnitude) eigenvalue for a given lambda_x and
    lambda_y."""

    self.Build_transport_matrix(lambda_x,lambda_y)
    if self.preconditioner==False :
      eigenvalues = scipy.linalg.eig(self.transport_matrix)
    else :
      mip = MIP.MIP(self.dof_handler,self.quad,self.sigma_t,self.sigma_s,
          lambda_x,lambda_y)
      mip.Build_matrix()
      mip.Invert()
      identity = np.eye(4*self.dof_handler.n_cells*self.quad.n_mom)
      matrix = self.transport_matrix+np.dot(mip.preconditioner,\
          self.transport_matrix-identity)
      eigenvalues = scipy.linalg.eig(matrix)
    tmp = np.zeros_like(eigenvalues[0])
    for i in xrange(0,eigenvalues[0].shape[0]) :
      tmp[i] = np.sqrt(eigenvalues[0][i].real**2+eigenvalues[0][i].imag**2)
    eigenvalue = tmp.max()
     
    return eigenvalue

#----------------------------------------------------------------------------#

  def Compute_condition_number(self) :
    """Compute the condition number. Assumes that Compute_largest_eigenvalue
    has been called before."""

    condition_number = np.linalg.cond(self.transport_matrix.toarray())
    return condition_number

#----------------------------------------------------------------------------#

  def Build_transport_matrix(self,lambda_x,lambda_y) :
    """Build the transport matrix for a given lambda_x and lambda_y."""

    self.lambda_x = lambda_x
    self.lambda_y = lambda_y
    m_size = 4*self.dof_handler.n_cells*self.quad.n_mom
    d_size = 4*self.dof_handler.n_cells*self.quad.n_dir
    self.L = np.zeros((d_size,d_size),dtype=complex)
    self.Build_scattering_matrix()

    for idir in xrange(0,self.quad.n_dir) :
# Direction alias
      omega_x = self.quad.omega[idir,0]
      omega_y = self.quad.omega[idir,1]

      for cell in self.dof_handler.grid :
        pos = 4*cell.fe_id+4*self.dof_handler.n_cells*idir
        self.L[pos:pos+4,pos:pos+4] += -omega_x*cell.x_grad_matrix-\
            omega_y*cell.y_grad_matrix+self.sigma_t*cell.mass_matrix

# Compute the normal of each side
        x = cell.y[1]-cell.y[0]
        y = cell.x[0]-cell.x[1]
        norm = np.sqrt(x**2+y**2)
        b_normal = np.array([x/norm,y/norm])
        x = cell.y[2]-cell.y[1]
        y = cell.x[1]-cell.x[2]
        norm = np.sqrt(x**2+y**2)
        r_normal = np.array([x/norm,y/norm])
        x = cell.y[3]-cell.y[2]
        y = cell.x[2]-cell.x[3]
        norm = np.sqrt(x**2+y**2)
        t_normal = np.array([x/norm,y/norm])
        x = cell.y[0]-cell.y[3]
        y = cell.x[3]-cell.x[0]
        norm = np.sqrt(x**2+y**2)
        l_normal = np.array([x/norm,y/norm])
        
        omega = np.array([omega_x,omega_y])

# Bottom term            
        n_dot_omega = np.dot(omega,b_normal)
# Upwind
        if n_dot_omega<0.0 :
          if cell.y[0]==self.dof_handler.bottom :
            offset = 4*self.dof_handler.nx_cells*\
                (self.dof_handler.ny_cells-1)
            self.Phase(cell,self.dof_handler.grid[cell.fe_id+\
                (self.dof_handler.ny_cells-1)*self.dof_handler.nx_cells],
                "bottom")
          else :
            offset = -4*self.dof_handler.nx_cells
            self.Phase(cell,self.dof_handler.grid[cell.fe_id-\
                self.dof_handler.nx_cells],"bottom")
          self.L[pos:pos+4,pos+offset:pos+offset+4] += n_dot_omega*\
              np.dot(cell.bottom_up,self.phase)
# Downwind 
        else :
          self.L[pos:pos+4,pos:pos+4] += n_dot_omega*cell.bottom_down

# Right term
        n_dot_omega = np.dot(omega,r_normal)
# Upwind
        if n_dot_omega<0.0 :
          if cell.x[1]==self.dof_handler.right :
            offset = -4*(self.dof_handler.nx_cells-1)
            self.Phase(cell,self.dof_handler.grid[cell.fe_id-\
                (self.dof_handler.nx_cells-1)],"right")
          else :
            offset = 4
            self.Phase(cell,self.dof_handler.grid[cell.fe_id+1],"right")
          self.L[pos:pos+4,pos+offset:pos+offset+4] += n_dot_omega*\
              np.dot(cell.right_up,self.phase)
# Downwind
        else :
          self.L[pos:pos+4,pos:pos+4] += n_dot_omega*cell.right_down
# Top term
        n_dot_omega = np.dot(omega,t_normal)  
# Upwind
        if n_dot_omega<0.0 :
          if cell.y[3]==self.dof_handler.top :
            offset = -4*self.dof_handler.nx_cells*\
                (self.dof_handler.ny_cells-1)
            self.Phase(cell,self.dof_handler.grid[cell.fe_id-\
                (self.dof_handler.ny_cells-1)*self.dof_handler.nx_cells],"top")
          else :
            offset = 4*self.dof_handler.nx_cells
            self.Phase(cell,self.dof_handler.grid[cell.fe_id+\
                self.dof_handler.nx_cells],"top")
          self.L[pos:pos+4,pos+offset:pos+offset+4] += n_dot_omega*\
              np.dot(cell.top_up,self.phase)
# Downwind 
        else :
          self.L[pos:pos+4,pos:pos+4] += n_dot_omega*cell.top_down
# Left term
        n_dot_omega = np.dot(omega,l_normal)
# Upwind
        if n_dot_omega<0.0 :
          if cell.x[0]==self.dof_handler.left :
            offset = 4*(self.dof_handler.nx_cells-1)
            self.Phase(cell,self.dof_handler.grid[cell.fe_id+\
                (self.dof_handler.nx_cells-1)],"left")
          else :
            offset = -4
            self.Phase(cell,self.dof_handler.grid[cell.fe_id-1],"left")
          self.L[pos:pos+4,pos+offset:pos+offset+4] += n_dot_omega*\
              np.dot(cell.left_up,self.phase)
# Downwind
        else :
          self.L[pos:pos+4,pos:pos+4] += n_dot_omega*cell.left_down

    identity = np.eye(4*self.dof_handler.n_cells)
    if self.solver_type=="SI" :
      D = scipy.sparse.kron(self.quad.D,identity)
      D = D.toarray()
      self.transport_matrix = np.dot(D,np.dot(scipy.linalg.inv(self.L),
        self.scattering_matrix))
    else :
      D = scipy.sparse.kron(self.quad.D,identity)
      D = D.toarray()
      self.transport_matrix = np.eye(4*self.dof_handler.n_cells*\
          self.quad.n_moments)-np.dot(D,np.dot(scipy.linalg.inv(self.L),
            self.scattering_matrix))
         
#----------------------------------------------------------------------------#

  def Build_scattering_matrix(self) :
    """Build the part of the scattering matrix."""

    m_unknowns = 4*self.quad.n_mom
    d_unknowns = 4*self.quad.n_dir
    m_size = self.dof_handler.n_cells*m_unknowns
    d_size = self.dof_handler.n_cells*d_unknowns
    matrix = np.zeros((m_size,m_size),dtype=complex)
    self.scattering_matrix = np.zeros((d_size,m_size),dtype=complex)
    for cell in self.dof_handler.grid :
      for mom in xrange(0,self.quad.n_mom) :
        pos = 4*mom+4*cell.fe_id*self.quad.n_mom
        matrix[pos:pos+4,pos:pos+4] = self.sigma_s[mom]*cell.mass_matrix
    identity = np.eye(4*self.dof_handler.n_cells)
    M = scipy.sparse.kron(self.quad.M,identity)
    M = M.toarray()
    self.scattering_matrix = np.dot(M,matrix)

#----------------------------------------------------------------------------#

  def Phase(self,cell_1,cell_2,side) :
    """Compute the phase matrix for a given matrix."""

    delta_x = np.zeros(4)
    delta_y = np.zeros(4)
    self.phase = np.zeros((4,4),dtype=complex)   
    if side=="bottom" :
      delta_x[2] = cell_2.x[2]-cell_1.x[1]
      delta_x[3] = cell_2.x[3]-cell_1.x[0]
      delta_y[2] = cell_2.y[2]-cell_1.y[1]
      delta_y[3] = cell_2.y[3]-cell_1.y[0]
    elif side=="right" :
      delta_x[0] = cell_2.x[0]-cell_1.x[1]
      delta_x[3] = cell_2.x[3]-cell_1.x[2]
      delta_y[0] = cell_2.y[0]-cell_1.y[1]
      delta_y[3] = cell_2.y[3]-cell_1.y[2]
    elif side=="top" :
      delta_x[0] = cell_2.x[0]-cell_1.x[3]
      delta_x[1] = cell_2.x[1]-cell_1.x[2]
      delta_y[0] = cell_2.y[0]-cell_1.y[3]
      delta_y[1] = cell_2.y[1]-cell_1.y[2]
    elif side=="left" :
      delta_x[1] = cell_2.x[1]-cell_1.x[0]
      delta_x[2] = cell_2.x[2]-cell_1.x[3]
      delta_y[1] = cell_2.y[1]-cell_1.y[0]
      delta_y[2] = cell_2.y[2]-cell_1.y[3]
    else :
      utils.Abort("This side does not exist")

    self.phase[0,0] = np.exp((self.lambda_x*delta_x[0]+self.lambda_y*delta_y[0])*1j)
    self.phase[1,1] = np.exp((self.lambda_x*delta_x[1]+self.lambda_y*delta_y[1])*1j)
    self.phase[2,2] = np.exp((self.lambda_x*delta_x[2]+self.lambda_y*delta_y[2])*1j)
    self.phase[3,3] = np.exp((self.lambda_x*delta_x[3]+self.lambda_y*delta_y[3])*1j)
