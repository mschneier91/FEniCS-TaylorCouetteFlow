#Michael Schneier's code
#Some changes thanks to Michael McLaughlin
#Some other changes due to Kiera Kean

from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import pdb
from os import sys


#Solver 
MAX_ITERS = 10000
solver = PETScKrylovSolver('gmres','none')
solver.ksp().setGMRESRestart(MAX_ITERS)
solver.parameters["monitor_convergence"] = False
solver.parameters['relative_tolerance'] = 1e-6
solver.parameters['maximum_iterations'] = MAX_ITERS
#Pi to high accuracy
Pi = np.pi

#DOF/timestep size
N = 40 
dt = .005

#Viscosity
nu = .001

#Final angular velocity of inner and outer cylinders
omega_inner = 1.391
omega_outer = 0
#Time it takes to reach full speed of cylinders
x = 1

#Problem timing
t_init = 0.0
t_final = 20
t_num = int((t_final - t_init)/dt)
t = t_init

#Define Domain
outer_top_radius =1.2
inner_top_radius = .7
outer_bot_radius = outer_top_radius
inner_bot_radius = inner_top_radius

b_ox = 0.
b_oy = 0.
b_oz = 0.

t_ox = 0.
t_oy = 0.
t_oz = 2.2

b_ix = 0.
b_iy = 0.
b_iz = 0.

t_ix = 0.
t_iy = 0.
t_iz = 2.2

domain = Cylinder(Point(b_ox, b_oy, b_oz), Point(t_ox, t_oy, t_oz), outer_top_radius, outer_bot_radius) - Cylinder(Point(b_ix, b_iy, b_iz), Point(t_ix, t_iy, t_iz), inner_top_radius, inner_bot_radius)

#Construct Mesh
meshrefine = 0 # 0,1, or 2: how many times boundary is refined
refine1 = .03 #distance from wall refined first time
refine2 = .01 #distance from wall refined second time

#Generate mesh for given cylinders
mesh = generate_mesh ( domain, N )

if (meshrefine > .5): #mesh refine greater than zero: refine once
    sub_domains_bool = MeshFunction("bool",mesh,mesh.topology().dim() - 1)
    sub_domains_bool.set_all(False)

    class SmallInterest(SubDomain):
        def inside(self, x, on_boundary):
            return (x[1]**2 + x[0]**2 < (inner_top_radius+refine1)**2 or x[1]**2 + x[0]**2 > (outer_top_radius-refine1)**2)


    interest = SmallInterest()
    interest.mark(sub_domains_bool,True)
    mesh = refine(mesh,sub_domains_bool)
    print("mesh refined")

    if (meshrefine > 1.5): #Greater than 2, refine a second time
        class BigInterest(SubDomain):
            def inside(self, x, on_boundary):
                return (x[1]**2 + x[0]**2 < (inner_top_radius+refine2)**2 or x[1]**2 + x[0]**2 > (outer_top_radius-refine2)**2)
        #
        sub_domains_bool2 = MeshFunction("bool",mesh,mesh.topology().dim() - 1)
        sub_domains_bool2.set_all(False)
        interest = BigInterest()
        interest.mark(sub_domains_bool2,True)
        mesh = refine(mesh,sub_domains_bool2)
        print("mesh refined again")

        

#Reynolds number info
height = t_iz - b_iz #hieght of the cylinder
radius_ratio = inner_bot_radius/outer_bot_radius #eta
d =  outer_bot_radius - inner_bot_radius 
aspect_ratio =  height/d 
domain_volume = aspect_ratio*d*2*Pi 

Re = d*omega/nu #Reynold's number as defined in Bilson/Bremhorst
Ta = 4*Re*Re*(1-radius_ratio)/(1+radius_ratio)

#Paraview Setup
savespersec = 20. #How many snapshots are taken per second
velocity_paraview_file = File("plot_N_"+str(N)+"_dt_"+str(dt)+"_omega_"+str(omega)+"_refine_"+str(meshrefine)+"/3d_TC_"+str(N)+"_"+str(dt)+"_"+str(omega)+".pvd")
snapspersec = int(1./dt) #timesteps per second
frameRate = int(snapspersec/savespersec) # how often snapshots are saved to paraview 
if frameRate ==0: #ensure no issues
    frameRate =1
    
           
    

#Print statements to ensure running correct version
print("N equals " + str(N))
print("dt equals " + str(dt))
print("omega equals " + str(omega))

#Print the total number of degrees of freedom
X_test = VectorFunctionSpace(mesh,"CG",2)
Q_test = FunctionSpace(mesh,"CG",1)
vdof = X_test.dim()
pdof = Q_test.dim()
print("The number of velocity DOFs is:" + str(vdof))
print("The number of pressure DOFs is:" + str(pdof))

print("The Reynolds number is " + str(Re))
print("The Taylor number is " + str(Ta))


#Taylor Hood element creation
V_h = VectorElement("Lagrange", mesh.ufl_cell(), 2) #Velocity space
Q_h = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #Pressure space
K_h = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #TKE space

W = FunctionSpace(mesh,MixedElement([V_h,Q_h]),constrained_domain=PeriodicBoundary())
K = FunctionSpace(mesh,K_h,constrained_domain=PeriodicBoundary())


#Set up trial and test functions for all parts
(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)
k = TrialFunction(K)
phi = TestFunction(K)

#Solution vectors
w_ = Function(W)
wnMinus1 = Function(W)
(unMinus1,pnMinus1) = wnMinus1.split(True) #Velocity solution vector at time n
wn = Function(W)
(un,pn) = wn.split(True) #Velocity solution vector at time n
wnPlus1 = Function(W)
(unPlus1,pnPlus1) = wnPlus1.split(True) #Velocity solution vector at time n+1
k_ = Function(K)
kn = Function(K) #TKE solution vector at time n
knPlus1 = Function(K) #TKE solution vector at time n+1



#Boundary Conditions: Setup and Definition
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

#Sub domain for the inner cylinder
class Inner_cyl(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]**2 + x[1]**2 <= inner_top_radius**2 +10**-2 and on_boundary

#Sub domain for the outer cylinder
class Outer_cyl(SubDomain):
    def inside(self, x, on_boundary):
        return outer_top_radius**2 - 10**-2 <= x[0]**2 + x[1]**2 and on_boundary

#Specify a point on the boundary for the pressures
mesh_points = mesh.coordinates()
class OriginPoint(SubDomain):
    def inside(self, x, on_boundary):
        tol = .001
        return (near(x[0], mesh_points[0,0])  and  near(x[1], mesh_points[0,1]) and near(x[2], mesh_points[0,2]) )
# Create mesh functions over the cell facets
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

# Mark all facets as sub domain 0
sub_domains.set_all(0)


#Mark the inner cylinder as sub domain 1
inner_cyl = Inner_cyl()
inner_cyl.mark(sub_domains, 1)

#Mark the outer cylinder as sub domain 2
outer_cyl= Outer_cyl()
outer_cyl.mark(sub_domains, 2)


#Smooth bridge for increasing inner angular velocity
def smooth_bridge(t):
    s = t/x
    #Smoothly increase from 0 at t=0 to 1 at t=s
    if(s>1+1e-14):
        return 1.0
    elif(abs(1-s)>1e-14):
        return np.exp(-np.exp(-1./(1-s)**2)/s**2)
    else:
        return 1.0
mint_val = smooth_bridge(t)


#Define boundary conditions for the velocity equation
noslip_u_outer = Expression(("mint*omega*r*x[1]/r", "mint*omega*r*-1*x[0]/r","0.0"), mint = 0.0,degree=4, r = outer_bot_radius,omega = omega) 
noslip_u_inner = Expression(("mint*omega*r*x[1]/r", "mint*omega*r*-1*x[0]/r","0.0"), mint = 0.0,degree=4, r = inner_bot_radius,omega = omega)

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
unMinus1.assign(Constant((0.0,0.0,0.0)))
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


#u euqation: 
eps=1.e-10 #Here to reduce problem to NSE
#Parameters for model
mu = .0000055
tau = .0000001
u_lhs = (1./dt)*inner(u,v)*dx + b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx + eps*np.sqrt(2.)*mu*tau*kn*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx 
u_rhs = (1./dt)*(inner(un,v))*dx

#k equation
k_lhs = (1./dt)*inner(k,phi)*dx + (nu+np.sqrt(2.)*mu*kn*tau)*a(k,phi)*dx + b(unPlus1,k,phi)*dx + (1./(np.sqrt(2.0)*tau))*inner(k,phi)*dx
k_rhs = (1./dt)*inner(kn,phi)*dx + np.sqrt(2.0)*mu*tau*kn*inner(.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T),.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T))*phi*dx #Last term here is the square of the symmetric gradient write function for this later


Au = None
Ak = None
Bu = None
Bk = None




np.savetxt('rs.txt',r)


for jj in range(0,t_num):
    t = t + dt
    print('Numerical Time Level: t = '+ str(t))

    #Matrix Assembly for the u equation
    Au = assemble(u_lhs)
    Bu = assemble(u_rhs)
    mint_val = smooth_bridge(t)
    noslip_u_outer.mint = mint_val

    #Application of boundary conditions for the u equation
    [bc.apply(Au,Bu) for bc in bcs_u]
    #Solve
    solver.solve(Au,w_.vector(),Bu)
    #Solution of the u equation
    (unPlus1,pnPlus1) = w_.split(True)
    #Apply time filter to correct dissipation in BE
    unPlus1.vector()[:] = unPlus1.vector()[:] -(1./3.)*(unPlus1.vector()[:]-2*un.vector()[:]+unMinus1.vector()[:])

    #Matrix Assembly for the k equation
    Ak = assemble(k_lhs)
    Bk = assemble(k_rhs)

    #Application of boundary conditions for the k equation
    [bc.apply(Ak,Bk) for bc in bcs_k]
    #Solve
    solve(Ak,k_.vector(),Bk)
    #Solution of the u equation
    knPlus1.assign(k_)


    if(jj%frameRate == 0):
        velocity_paraview_file << (unPlus1,t)
    
    ###Following code will save true values along some select points to compare with analytic solution.
#    if abs(t-10)<1.e-6:
        #thetaPoints = 2 #number of points we take along varying theta 
        #zpoints = 2 #number of points we take along z axis
        #rpoints = 100 #points going out radially
        #
        #r = np.zeros((rpoints))
        #theta = np.zeros((thetaPoints))
        #z = np.zeros((zpoints))
        #
        #
        #for i in range(0,thetaPoints):
        #    theta[i] = i*(2*Pi)/thetaPoints
        #for i in range(0,zpoints):
        #    z[i] = b_iz+t_iz*(i+1)/(zpoints+1)
        #for i in range(0,rpoints):
        #    r[i] = inner_top_radius+(outer_top_radius-inner_top_radius)*(i+1)/(rpoints+2)
#        for i in range(0,thetaPoints):
#            for j in range(0,zpoints):
#                u_r_vals = np.zeros(rpoints)
#                u_th_vals = np.zeros(rpoints)
#                u_z_vals = np.zeros(rpoints)
#                u_x_vals = np.zeros(rpoints)
#                u_y_vals = np.zeros(rpoints)
#                p_vals = np.zeros(rpoints)
#                for k in range(0,rpoints):
#                    # print("r "+str(r[k]))
#                    # print("theta "+str(cos(theta[i])))
#                    # print("x "+str(r[k]*cos(theta[i])))
                    # print("y "+str(r[k]*sin(theta[i])))
                    # print("z "+str(z[j]))
#                    uvw = unPlus1(r[k]*cos(theta[i]), r[k]*sin(theta[i]),z[j])
#                    u_x_vals[k] = uvw[0]
#                    u_y_vals[k] = uvw[1]
#                    u_r_vals[k] = uvw[0]*cos(theta[i])+uvw[1]*sin(theta[i])
#                    u_th_vals[k]= uvw[1]*cos(theta[i])-uvw[0]*sin(theta[i])
#                    u_z_vals[k]= uvw[2]
#                    p_vals[k] = pnPlus1(r[k]*cos(theta[i]), r[k]*sin(theta[i]),z[j])
#
##                plt.figure(1)
##                plt.plot(r,u_r_vals,"b",r,u_th_vals,"k",r,u_z_vals,"r", label=r"velocity plots",linewidth =.5 )
##                plt.xlabel("r")
##                plt.ylabel("velocity_"+str(omega))
##
##                plt.savefig("N_" + str(N) +"_i_"+str(i)+"_j_"+str(j))
##                plt.close()
#                
#                np.savetxt("N_" + str(N) + "omega"+str(omega)+"_i_"+str(i)+"_j_"+str(j)+"pvals.txt", p_vals)
#                np.savetxt("N_" + str(N) + "omega"+str(omega)+"_i_"+str(i)+"_j_"+str(j)+"Uth.txt", u_th_vals)
#                np.savetxt("N_" + str(N) + "omega"+str(omega)+"_i_"+str(i)+"_j_"+str(j)+"Ur.txt", u_r_vals)
#                np.savetxt("N_" + str(N) + "omega"+str(omega)+"_i_"+str(i)+"_j_"+str(j)+"Uz.txt", u_z_vals)
#        
    #Assign values for next time step
    unMinus1.assign(un)
    un.assign(unPlus1)
    kn.assign(knPlus1)

