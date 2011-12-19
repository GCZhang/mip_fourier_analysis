# Python code
# Author: Bruno Turcksin
# Date: 2011-11-28 10:25:49.101948

#----------------------------------------------------------------------------#
## Class MIP                                                                ##
#----------------------------------------------------------------------------#

"""Invert the MIP equation"""

import numpy as np
import utils

class MIP(object) :
  """Preconditioner for the transport using the MIP equation."""

  def __init__(self,dof_handler,quad,sigma_t,sigma_s,lambda_x,lambda_y) :

    super(MIP,self).__init__()
    self.dof_handler = dof_handler
    self.sigma_t = sigma_t
    self.sigma_s = sigma_s
    self.sigma_a = self.sigma_t-self.sigma_s[0]
    self.quad = quad
    self.lambda_x = lambda_x
    self.lambda_y = lambda_y
    self.edges = ['bottom','right','top','left']

#----------------------------------------------------------------------------#

  def Invert(self) :
    """Invert the MIP equation."""

    scalar_flux_size = 4*self.dof_handler.n_cells
    unknowns = scalar_flux_size*self.quad.n_mom
    restriction = np.zeros((scalar_flux_size,unknowns),dtype='complex')
    projection = np.zeros((unknowns,scalar_flux_size),dtype='complex')
    b = np.zeros((scalar_flux_size,scalar_flux_size),dtype='complex')

    for cell in self.dof_handler.grid :
      fe_id = cell.fe_id
      b[4*fe_id:4*fe_id+4,4*fe_id:4*fe_id+4] += self.sigma_s[0]*\
          cell.mass_matrix
      for i in xrange(0,4) :
        restriction[i,i+fe_id*4*self.quad.n_mom] = 1.
        projection[i+fe_id*4*self.quad.n_mom,i] = 1.

    invert = np.linalg.inv(self.mip_matrix)
    alpha = np.dot(b,restriction)
    beta = np.dot(invert,alpha)
    gamma = np.dot(projection,beta)
    self.preconditioner = gamma
   # if self.lambda_x==0 and self.lambda_y==0 :
   #   print invert
   #   print self.preconditioner

#----------------------------------------------------------------------------#

  def Build_matrix(self) :
    """Build the matrix of the MIP that will be inverted."""

    n_unknowns = 4*self.dof_handler.n_cells
    self.mip_matrix = np.zeros((n_unknowns,n_unknowns),dtype='complex')

    for cell in self.dof_handler.grid :
      if self.sigma_s.shape[0]>1 :
        sigma_tr = self.sigma_t - self.sigma_s[1]
      else :
        sigma_tr = self.sigma_t
      D = 1./(3.*sigma_tr)
      D_m = D
      D_p = D
      pos = 4*cell.fe_id

      self.mip_matrix[pos:pos+4,pos:pos+4] += self.sigma_a*cell.mass_matrix
      self.mip_matrix[pos:pos+4,pos:pos+4] += D*cell.stiffness_matrix
     # if self.lambda_x==0 and self.lambda_y==0 :
     #   print self.mip_matrix

# Loop over the edge of this cell
      for edge in self.edges :
        inside = self.Is_interior(cell,edge)
        K = self.Compute_penalty_coefficient(cell,edge,D) 
        if inside==True :
          if edge=='left' or edge=='bottom' :
            if edge=='left' :
              edge_deln_matrix = cell.edge_deln_matrix['left']
              in_across_edge_deln_matrix = cell.edge_deln_matrix['left']
              out_across_edge_deln_matrix = cell.edge_deln_matrix['right']
              edge_mass = cell.left_edge_mass_matrix
              coupling_edge_mass = cell.left_coupling_edge_mass_matrix
              offset = -4
              offset_id = -1
              i1 = [0,3]
              j1 = [0,3]
              j2 = [1,2]
            else :
              edge_deln_matrix = cell.edge_deln_matrix['bottom']
              in_across_edge_deln_matrix = cell.edge_deln_matrix['bottom']
              out_across_edge_deln_matrix = cell.edge_deln_matrix['top']
              edge_mass = cell.bottom_edge_mass_matrix
              coupling_edge_mass = cell.bottom_coupling_edge_mass_matrix
              offset = -4*self.dof_handler.nx_cells
              offset_id = -self.dof_handler.nx_cells
              i1 = [0,1]
              j1 = [0,1]
              j2 = [2,3]
# Internal term (+,+)
            self.mip_matrix[pos:pos+4,pos:pos+4] += 0.5*D_m*\
                edge_deln_matrix
            self.mip_matrix[pos:pos+4,pos:pos+4] += 0.5*D_m*\
                edge_deln_matrix.transpose()
# Mixte term (-,+)
            self.Phase(cell,self.dof_handler.grid[cell.fe_id+offset_id],edge)
            self.mip_matrix[pos:pos+4,pos+offset:pos+offset+4] -= 0.5*D_m*\
                np.dot(out_across_edge_deln_matrix,self.phase)
            self.mip_matrix[pos:pos+4,pos+offset:pos+offset+4] += 0.5*D_p*\
                np.dot(in_across_edge_deln_matrix,self.phase)
          else :
            if edge=='right' :
              edge_deln_matrix = cell.edge_deln_matrix['right']
              in_across_edge_deln_matrix = cell.edge_deln_matrix['left']
              out_across_edge_deln_matrix = cell.edge_deln_matrix['right']
              edge_mass = cell.right_edge_mass_matrix
              coupling_edge_mass = cell.right_coupling_edge_mass_matrix
              offset = 4
              offset_id = 1
              i1 = [1,2]
              j1 = [1,2]
              j2 = [0,3]
            else :
              edge_deln_matrix = cell.edge_deln_matrix['top']
              in_across_edge_deln_matrix = cell.edge_deln_matrix['bottom']
              out_across_edge_deln_matrix = cell.edge_deln_matrix['top']
              edge_mass = cell.top_edge_mass_matrix
              coupling_edge_mass = cell.top_coupling_edge_mass_matrix
              offset = 4*self.dof_handler.nx_cells
              offset_id = self.dof_handler.nx_cells
              i1 = [2,3]
              j1 = [2,3]
              j2 = [0,1]
# Internal term (-,-)
            self.mip_matrix[pos:pos+4,pos:pos+4] -= 0.5*D_m*\
                edge_deln_matrix
            self.mip_matrix[pos:pos+4,pos:pos+4] -= 0.5*D_m*\
                edge_deln_matrix.transpose()
# Mixte term (+,-)
            self.Phase(cell,self.dof_handler.grid[cell.fe_id+offset_id],edge)
            self.mip_matrix[pos:pos+4,pos+offset:pos+offset+4] += 0.5*D_m*\
                np.dot(in_across_edge_deln_matrix,self.phase)
            self.mip_matrix[pos:pos+4,pos+offset:pos+offset+4] -= 0.5*D_p*\
                np.dot(out_across_edge_deln_matrix,self.phase)
# First edge term
          for i in xrange(0,2) :
            for j in xrange(0,2) :
              self.mip_matrix[pos+i1[i],pos+j1[j]] += K*edge_mass[i,j]
              self.mip_matrix[pos+i1[i],pos+offset+j2[j]] -=\
                  K*coupling_edge_mass[i][j]
        else :
          if edge=='right' or edge=='top' :
            if edge=='right' :
              opposite_edge = 'left'
              edge_mass_matrix = cell.right_edge_mass_matrix
              coupling_edge_mass_matrix = cell.right_coupling_edge_mass_matrix
              edge_deln_matrix = cell.edge_deln_matrix['right']
              outside_edge_deln_matrix = cell.edge_deln_matrix['left']
              in_across_edge_deln_matrix =\
                  cell.across_edge_deln_matrix['right']
              out_across_edge_deln_matrix =\
                  cell.across_edge_deln_matrix['left']
              offset = -4*(self.dof_handler.nx_cells-1)
              offset_id = -(self.dof_handler.nx_cells-1)
              i1 = [1,2]
              j1 = [1,2]
              i2 = [0,3]
              j2 = [0,3]
            else :
              opposite_edge = 'bottom'
              edge_mass_matrix = cell.top_edge_mass_matrix
              coupling_edge_mass_matrix = cell.top_coupling_edge_mass_matrix
              edge_deln_matrix = cell.edge_deln_matrix['top']
              outside_edge_deln_matrix = cell.edge_deln_matrix['bottom']
              in_across_edge_deln_matrix =\
                  cell.across_edge_deln_matrix['top']
              out_across_edge_deln_matrix =\
                  cell.across_edge_deln_matrix['bottom']
              offset = -4*self.dof_handler.nx_cells*\
                  (self.dof_handler.ny_cells-1)
              offset_id = -self.dof_handler.nx_cells*\
                  (self.dof_handler.ny_cells-1)
              i1 = [2,3]
              j1 = [2,3]
              i2 = [0,1]
              j2 = [0,1]
# First edge term
            for i in xrange(0,2) :
              for j in xrange(0,2) :
                self.mip_matrix[pos+i1[i],pos+j1[j]] += K*\
                    edge_mass_matrix[i,j]
                self.mip_matrix[pos+offset+i2[i],pos+offset+j2[j]] +=\
                    K*edge_mass_matrix[i,j]
                self.Phase(self.dof_handler.grid[cell.fe_id+offset_id],
                    cell,opposite_edge)
                self.mip_matrix[pos+offset+i2[i],pos+j1[j]] -= K*\
                    coupling_edge_mass_matrix[i,j]*self.phase[j1[j],j1[j]]
                self.Phase(cell,self.dof_handler.grid[cell.fe_id+offset_id],
                    edge)
                self.mip_matrix[pos+i1[i],pos+offset+j2[j]] -= K*\
                    coupling_edge_mass_matrix[i,j]*self.phase[j2[j],j2[j]]
# Internal terms (-,-)
            self.mip_matrix[pos:pos+4,pos:pos+4] -= 0.5*D_m*\
                edge_deln_matrix
            self.mip_matrix[pos:pos+4,pos:pos+4] -= 0.5*D_m*\
                edge_deln_matrix.transpose()
# Mixte terms (-,+)
            self.Phase(self.dof_handler.grid[cell.fe_id+offset_id],cell,
                opposite_edge)
            self.mip_matrix[pos+offset:pos+offset+4,pos:pos+4] += 0.5*D_m*\
                np.dot(in_across_edge_deln_matrix.transpose(),self.phase)
            self.mip_matrix[pos+offset:pos+offset+4,pos:pos+4] -= 0.5*D_p*\
                np.dot(out_across_edge_deln_matrix,self.phase)
# External terms (+,+)
            self.mip_matrix[pos+offset:pos+offset+4,pos+offset:pos+offset+4] +=\
                0.5*D_p*outside_edge_deln_matrix
            self.mip_matrix[pos+offset:pos+offset+4,pos+offset:pos+offset+4] +=\
                0.5*D_p*outside_edge_deln_matrix.transpose()
# Mixte terms (+,-)
            self.Phase(cell,self.dof_handler.grid[cell.fe_id+offset_id],edge)
            self.mip_matrix[pos:pos+4,pos+offset:pos+offset+4] += 0.5*D_m*\
                np.dot(in_across_edge_deln_matrix,self.phase)
            self.mip_matrix[pos:pos+4,pos+offset:pos+offset+4] -= 0.5*D_p*\
                np.dot(out_across_edge_deln_matrix.transpose(),self.phase)

#----------------------------------------------------------------------------#

  def Is_interior(self,cell,edge) :
    """Return false if the edge is on the boundary of the domain."""

    value = True
    if edge=='left' and cell.x[0]==self.dof_handler.left :
      value = False
    elif edge=='right' and cell.x[1]==self.dof_handler.right :
      value = False
    elif edge=='bottom' and cell.y[0]==self.dof_handler.bottom :
      value = False
    elif edge=='top' and cell.y[2]==self.dof_handler.top :
      value = False

    return value

#----------------------------------------------------------------------------#

  def Compute_penalty_coefficient(self,cell,edge,D) :
    """Compute the penalty coefficient for a given edge."""

    nx = self.dof_handler.nx_cells
    ny = self.dof_handler.ny_cells
    if edge=='left' :
      edge_length = np.sqrt((cell.x[3]-cell.x[0])**2+(cell.y[3]-cell.y[0])**2)
      if cell.x[0]==self.dof_handler.left :
        value = 2.*D*(cell.area/edge_length+\
            self.dof_handler.grid[cell.fe_id+nx-1].area/edge_length)
      else :
        value = 2.*D*(cell.area/edge_length+\
            self.dof_handler.grid[cell.fe_id-1].area/edge_length)
    elif edge=='right' :
      edge_length = np.sqrt((cell.x[2]-cell.x[1])**2+(cell.y[2]-cell.y[1])**2)
      if cell.x[1]==self.dof_handler.right :
        value = 2.*D*(cell.area/edge_length+\
            self.dof_handler.grid[cell.fe_id-nx+1].area/edge_length)
      else :
        value = 2.*D*(cell.area/edge_length+\
            self.dof_handler.grid[cell.fe_id+1].area/edge_length)
    elif edge=='bottom' :
      edge_length = np.sqrt((cell.x[1]-cell.x[0])**2+(cell.y[1]-cell.y[0])**2)
      if cell.y[0]==self.dof_handler.bottom :
        value = 2.*D*(cell.area/edge_length+\
            self.dof_handler.grid[cell.fe_id+nx*(ny-1)].area/edge_length)
      else :
        value = 2.*D*(cell.area/edge_length+\
            self.dof_handler.grid[cell.fe_id-nx].area/edge_length)
    else :
      edge_length = np.sqrt((cell.x[3]-cell.x[2])**2+(cell.y[3]-cell.y[2])**2)
      if cell.y[2]==self.dof_handler.top :
        value = 2.*D*(cell.area/edge_length+\
            self.dof_handler.grid[cell.fe_id-nx*(ny-1)].area/edge_length)
      else :
        value = 2.*D*(cell.area/edge_length+\
            self.dof_handler.grid[cell.fe_id+nx].area/edge_length)
    
    return np.max([value,0.25])

#----------------------------------------------------------------------------#

  def Phase(self,cell_1,cell_2,side) :
    """Compute the phase matrix."""

    delta_x = np.zeros(4)
    delta_y = np.zeros(4)
    self.phase = np.zeros((4,4),dtype=complex)   
    if side=="bottom" :
      delta_x[0] = cell_2.x[0]-cell_1.x[0]
      delta_y[0] = cell_2.y[0]-cell_1.y[0]
      delta_x[1] = cell_2.x[1]-cell_1.x[1]
      delta_y[1] = cell_2.y[1]-cell_1.y[1]
      
      delta_x[2] = delta_x[1]
      delta_x[3] = delta_x[0]
      delta_y[2] = delta_y[1]
      delta_y[3] = delta_y[0]

    #  delta_x[2] = cell_2.x[2]-cell_1.x[1]
    #  delta_x[3] = cell_2.x[3]-cell_1.x[0]
    #  delta_y[2] = cell_2.y[2]-cell_1.y[1]
    #  delta_y[3] = cell_2.y[3]-cell_1.y[0]
    elif side=="right" :
      delta_x[1] = cell_2.x[1]-cell_1.x[1]
      delta_y[1] = cell_2.y[1]-cell_1.y[1]
      delta_x[2] = cell_2.x[2]-cell_1.x[2]
      delta_y[2] = cell_2.y[2]-cell_1.y[2]

      delta_x[0] = delta_x[1]
      delta_x[3] = delta_x[2]
      delta_y[0] = delta_y[1]
      delta_y[3] = delta_y[2]

     # delta_x[0] = cell_2.x[0]-cell_1.x[1]
     # delta_x[3] = cell_2.x[3]-cell_1.x[2]
     # delta_y[0] = cell_2.y[0]-cell_1.y[1]
     # delta_y[3] = cell_2.y[3]-cell_1.y[2]
    elif side=="top" :
      delta_x[2] = cell_2.x[2]-cell_1.x[2]
      delta_y[2] = cell_2.y[2]-cell_1.y[2]
      delta_x[3] = cell_2.x[3]-cell_1.x[3]
      delta_y[3] = cell_2.y[3]-cell_1.y[3]
      
      delta_x[0] = delta_x[3]
      delta_x[1] = delta_x[2]
      delta_y[0] = delta_y[3]
      delta_y[1] = delta_y[2]

     # delta_x[0] = cell_2.x[0]-cell_1.x[3]
     # delta_x[1] = cell_2.x[1]-cell_1.x[2]
     # delta_y[0] = cell_2.y[0]-cell_1.y[3]
     # delta_y[1] = cell_2.y[1]-cell_1.y[2]
    elif side=="left" :
      delta_x[0] = cell_2.x[0]-cell_1.x[0]
      delta_y[0] = cell_2.y[0]-cell_1.y[0]
      delta_x[3] = cell_2.x[3]-cell_1.x[3]
      delta_y[3] = cell_2.y[3]-cell_1.y[3]

      delta_x[2] = delta_x[3]
      delta_x[1] = delta_x[0]
      delta_y[2] = delta_y[3]
      delta_y[1] = delta_y[0]

     # delta_x[1] = cell_2.x[1]-cell_1.x[0]
     # delta_x[2] = cell_2.x[2]-cell_1.x[3]
     # delta_y[1] = cell_2.y[1]-cell_1.y[0]
     # delta_y[2] = cell_2.y[2]-cell_1.y[3]
    else :
      utils.Abort("This side does not exist")

    self.phase[0,0] = np.exp((self.lambda_x*delta_x[0]+self.lambda_y*delta_y[0])*1j)
    self.phase[1,1] = np.exp((self.lambda_x*delta_x[1]+self.lambda_y*delta_y[1])*1j)
    self.phase[2,2] = np.exp((self.lambda_x*delta_x[2]+self.lambda_y*delta_y[2])*1j)
    self.phase[3,3] = np.exp((self.lambda_x*delta_x[3]+self.lambda_y*delta_y[3])*1j)
