from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import pdb
from os import sys


#Construct Mesh
outer_top_radius = 1.
outer_bot_radius = outer_top_radius
inner_top_radius = .5
inner_bot_radius = inner_top_radius
N = 25

#coordinates for defining the domain
b_ox = 0.
b_oy = 0.
b_oz = 0.

t_ox = 0.
t_oy = 0.
t_oz = np.pi/2.

b_ix = 0.
b_iy = 0.
b_iz = 0.

t_ix = 0.
t_iy = 0.
t_iz = np.pi/2.

height = t_iz - b_iz #hieght of the cylinder

domain = Cylinder(Point(b_ox, b_oy, b_oz), Point(t_ox, t_oy, t_oz), outer_top_radius, outer_bot_radius) - Cylinder(Point(b_ix, b_iy, b_iz), Point(t_ix, t_iy, t_iz), inner_top_radius, inner_bot_radius)
mesh = generate_mesh ( domain, N )

#Parameters
mu = .55
tau = .1
nu = .0001
omega = 2. #angular velocity

#Time info
t_init = 0.0
t_final = 5.0
dt = .01
t_num = int((t_final - t_init)/dt)
t = t_init

#Reynolds and Taylor number info
radius_ratio = inner_bot_radius/outer_bot_radius
d =  outer_bot_radius - inner_bot_radius
aspect_ratio =  height/d
rot_velocity = inner_top_radius*omega
Re = inner_bot_radius*d*omega/nu
Ta = Re**2 *(1. - 1./radius_ratio)


#Mark subdomains for bounadry conditions we use 

# class Top(SubDomain):
#     def inside(self, x, on_boundary):
#         return  t_oz - DOLFIN_EPS <= x[2] and on_boundary 

# #Sub domain for the bottom of the cylinder
# class Bottom(SubDomain):
#     def inside(self, x, on_boundary):
#         return x[2] <= b_iz + DOLFIN_EPS and on_boundary


# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Bottom of the cylinder is the "target domain" 
    def inside(self, x, on_boundary):
        return x[2] <= b_iz + DOLFIN_EPS and on_boundary 

    # Map top of the cylinder to the bottom of the cylinder 
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1]
        y[2] = x[2] - height

###########################################################
####BUG boundaires are probably marking everything!########
###########################################################
#Sub domain for the inner cylinder
class Inner_cyl(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]**2 + x[1]**2 <= inner_top_radius**2 + DOLFIN_EPS and on_boundary

#Sub domain for the outer cylinder 
#The tolerance had to be set much differently for this condition for some reason, double check this when swithching the mesh
class Outer_cyl(SubDomain):
    def inside(self, x, on_boundary):
        return outer_top_radius**2 - 10**-2 <= x[0]**2 + x[1]**2 and on_boundary

#Specify a point on the boundary for the pressures
#I need to check this later
mesh_points = mesh.coordinates()
class OriginPoint(SubDomain):
    def inside(self, x, on_boundary):
        tol = .001
        return (near(x[0], mesh_points[0,0])  and  near(x[1], mesh_points[0,1]) and near(x[2], mesh_points[0,2]) )
# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

# Mark all facets as sub domain 0
sub_domains.set_all(0)

# Mark the top of the cylinder as sub domain 1
# top = Top()
# top.mark(sub_domains, 1)

#Mark the bottom of the cylinder as sub domain 2
# bottom = Bottom()
# bottom.mark(sub_domains, 2)

#Mark the inner cylinder as sub domain 1
inner_cyl = Inner_cyl()
inner_cyl.mark(sub_domains, 1)


#Mark the outer cylinder as sub domain 2 
outer_cyl= Outer_cyl()
outer_cyl.mark(sub_domains, 2)

#Save sub domains and examine in paraview to check correct assignment, this can be finicky so make sure to check before a run
# file = File("subdomains.pvd")
# file << sub_domains



#Taylor Hood element creation
V_h = VectorElement("Lagrange", mesh.ufl_cell(), 2) #Velocity space
Q_h = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #Pressure space
K_h = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #TKE space

W = FunctionSpace(mesh,MixedElement([V_h,Q_h]),constrained_domain=PeriodicBoundary())    
K = FunctionSpace(mesh,K_h,constrained_domain=PeriodicBoundary())

X_test = VectorFunctionSpace(mesh,"CG",2)
Q_test = FunctionSpace(mesh,"CG",1)


#Print the total number of degrees of freedom
vdof = X_test.dim() 
pdof = Q_test.dim()
print("The number of velocity DOFs is:" + str(vdof))
print("The number of pressure DOFs is:" + str(pdof))

#Set up trial and test functions for all parts
(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)
k = TrialFunction(K)
phi = TestFunction(K)

#Solution vectors
w_ = Function(W)
wn = Function(W)
(un,pn) = wn.split(True) #Velocity solution vector at time n
wnPlus1 = Function(W)
(unPlus1,pnPlus1) = wnPlus1.split(True) #Velocity solution vector at time n+1
k_ = Function(K)
kn = Function(K) #TKE solution vector at time n
knPlus1 = Function(K) #TKE solution vector at time n+1

#Define boundary conditions for the velocity equation
noslip_u_inner = Expression(("rot_velocity*x[1]/r", "rot_velocity*-1*x[0]/r","0.0"), degree=2, r = inner_bot_radius,rot_velocity =rot_velocity)
noslip_u_outer = Constant((0.0, 0.0, 0.0))
originpoint = OriginPoint()
bc_inner= DirichletBC(W.sub(0),noslip_u_inner,sub_domains,1) #boundary condition for inner cylinder
bc_outer = DirichletBC(W.sub(0),noslip_u_outer,sub_domains,2) #boundary condition for outer cylinder
bcp = DirichletBC(W.sub(1), 0.0, originpoint, 'pointwise') #specify a point on the boundary for the pressure
bcs_u = [bc_outer,bc_inner,bcp]

#Define boundary conditions for the k equation
bc_inner_k = DirichletBC(K,0.0,sub_domains,1) #boundary condition for k on the inner cylinder
bc_outer_k = DirichletBC(K,0.0,sub_domains,2) #boundary condition for k on the outer cylinder
bcs_k = [bc_inner_k,bc_outer_k]



#Assign initial conditions (Start from rest)
un.assign(Constant((0.0,0.0,0.0)))
kn.assign(Constant(0.0))



#Weak Formulations
def a(u,v):                                                                                    
    return inner(nabla_grad(u),nabla_grad(v))
def a_sym(u,v):                                                                                    
    return inner(.5*(nabla_grad(u)+nabla_grad(u).T),nabla_grad(v))
def b(u,v,w):                                                                                      
    return .5*(inner(dot(u,nabla_grad(v)),w)-inner(dot(u,nabla_grad(w)),v))                         
def convect(u,v,w):
	return dot(dot(u, nabla_grad(v)), w) 
def c(p,v):
	return inner(p,div(v))



#u euqation
u_lhs = (1./dt)*inner(u,v)*dx + b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx + np.sqrt(2.)*mu*tau*kn*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx  
u_rhs = (1./dt)*(inner(un,v))*dx

#k equation
k_lhs = (1./dt)*inner(k,phi)*dx + (nu+np.sqrt(2.)*mu*kn*tau)*a(k,phi)*dx + b(unPlus1,k,phi)*dx + (1./(np.sqrt(2.0)*tau))*inner(k,phi)*dx
k_rhs = (1./dt)*inner(kn,phi)*dx + np.sqrt(2.0)*mu*tau*kn*inner(.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T),.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T))*phi*dx #Last term here is the square of the symmetric gradient write function for this later

#Filenames for saving plots
velocity_paraview_file = File("paraview_plotting/3d_Taylor_Couette.pvd")


#Arrays for holding statistics
nu_eff_arr = np.zeros(t_num)
VgradNorm[k_count-k_init] = np.zeros(t_num)

for jj in range(0,t_num):
    t = t + dt
    print('Numerical Time Level: t = '+ str(t))

    #Matrix Assembly for the u equation
    Au = assemble(u_lhs)
    Bu = assemble(u_rhs)

    #Application of boundary conditions for the u equation
    [bc.apply(Au,Bu) for bc in bcs_u]
    #Solve
    solve(Au,w_.vector(),Bu)
    #Solution of the u equation
    (unPlus1,pnPlus1) = w_.split(True)

    #Matrix Assembly for the k equation
    Ak = assemble(k_lhs)
    Bk = assemble(k_rhs)

    #Application of boundary conditions for the k equation
    [bc.apply(Ak,Bk) for bc in bcs_k]
    #Solve
    solve(Ak,k_.vector(),Bk)
    #Solution of the u equation
    knPlus1.assign(k_)

    #Save solution
    if(jj%10 == 0):
    #Save solution
        velocity_paraview_file << (unPlus1,t)



    #Calculate Turbulence Statistics
    VgradNorm[jj] = sqrt(assemble(inner(.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T),.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T))*dx))
    nu_eff_arr[jj] = assemble((nu+np.sqrt(2.0)*mu*knPlus1*tau)*inner(.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T),.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T))*dx)/pow(VgradNorm[jj],2.0)
    # Nu_Effective[k_count-k_init] = assemble((nu+np.sqrt(2.0)*mu*knPlus1*tau)*inner(.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T),.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T))*dx)/pow(VgradNorm[k_count-k_init],2.0)
    # nu_effective.write('%.10e&%.10e\n'%(t,Nu_Effective[k_count-k_init]))
    # Intensity_Model[k_count-k_init] = 2*assemble(knPlus1*dx)/pow(VNorm[k_count-k_init],2.0)
    # intensity_model.write('%.10e&%.10e\n'%(t,Intensity_Model[k_count-k_init]))
    # Viscosity_Ratio[k_count-k_init] = assemble(np.sqrt(2.0)*mu_2*knPlus1*tau*inner(.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T),.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T))*dx)/(2*nu*pow(VgradNorm[k_count-k_init],2.0))
    # viscosity_ratio.write('%.10e&%.10e\n'%(t,Viscosity_Ratio[k_count-k_init]))
    # Taylor_Microscale[k_count-k_init] = pow(pow(VgradNorm[k_count-k_init],2)/pow(VNorm[k_count-k_init],2),-.5)
    # taylor_microscale.write('%.10e&%.10e\n'%(t,Taylor_Microscale[k_count-k_init]))
    # Avg_l[k_count-k_init] = pow((1./(.99*np.pi))*assemble(2.0*tau*knPlus1*dx),.5)
    # avg_l.write('%.10e&%.10e\n'%(t,Avg_l[k_count-k_init]))
    # Avg_Turbulent_nu[k_count-k_init] = (1./(.99*np.pi))*assemble(np.sqrt(2.0)*mu_2*tau*knPlus1*dx)
    # avg_turbulent_nu.write('%.10e&%.10e\n'%(t,Avg_Turbulent_nu[k_count-k_init]))
​
    #Assign values for next time step
    un.assign(unPlus1)
    kn.assign(knPlus1)


    








