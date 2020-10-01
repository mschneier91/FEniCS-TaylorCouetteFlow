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


#Solver (this bit just copied from something online)
MAX_ITERS = 10000
solver = PETScKrylovSolver('gmres','none')
solver.ksp().setGMRESRestart(MAX_ITERS)
solver.parameters["monitor_convergence"] = False
solver.parameters['relative_tolerance'] = 1e-6
solver.parameters['maximum_iterations'] = MAX_ITERS
#solver.parameters['absolute_tolerance'] = 1.0e-9
# solver = KrylovSolver()


#####THINGS I HAVE BEEN CHANGING A LOT LATELY
#DOF/timestep size
N = 42
dt = .005
#Final speed (how we vary Taylor number)
omega = 3.2
#How fast we increase inner radius
x = 1. #That is, it takes x seconds to reach full speed
meshrefine = 0 # 0,1, or 2, how much we refine the mesh on the boundary, generally I just make this 1, 2 makes dof INSANE
refine1 = .05#how far in we refine the first time
refine2 = .02 #another refining thing
#Print out choices so we can see what is running
print("N equals " + str(N))
print("dt equals " + str(dt))
print("omega equals " + str(omega))
print("x equals " + str(x))
#Filenames for saving plots (So we can tell what version we're looking at and dont write over things on accident)
velocity_paraview_file = File("plot_"+str(dt)+"_"+str(N)+"_"+str(omega)+"_"+str(meshrefine)+"_bridge_"+str(x)+"/3d_TC_sb_"+str(dt)+"_"+str(N)+"_"+str(omega)+".pvd")
q_file = File("plot_"+str(dt)+"_"+str(N)+"_"+str(omega)+"_"+str(meshrefine)+"_bridge_"+str(x)+"/q.pvd")

#how often do we save to paraview?
snapspersec = int(1./dt) #this many timesteps per second
savespersec = 20. #we want to take this many snapshots per second
frameRate = int(snapspersec/savespersec) # so we take a snapshot this often
if frameRate ==0:#make sure not issues
    frameRate =1

#Time info
t_init = 0.0
t_final = x+15. #fifteen seconds after we reach full speed (Hopefully enough time to fully evolve or do whatever)
#(since we only output paraview stuff no real benefit to stopping early? In this version??)
t_final = 20
t_num = int((t_final - t_init)/dt)
t = t_init
tol = 1.0E-10
#Smooth bridge (to allow speed to increase slowly or not, set x < dt to basically ignore this part)
def smooth_bridge(t):
    s = t/x
    #we're at full speed at x seconds
    #This is definitely a dumb way to define it but it works with how I defined inner radius and I don't wanna debug more
    #I cannot trust my algebra in general
    #basically, sanity check s = 1 at t = x
    if(s>1+1e-14):
        return 1.0
    elif(abs(1-s)>1e-14):
        return np.exp(-np.exp(-1./(1-s)**2)/s**2)
    else:
        return 1.0




mint_val = smooth_bridge(t)


#Construct Mesh
outer_top_radius =2.611
outer_bot_radius = 2.611
inner_top_radius = 1.611
inner_bot_radius = 1.611


#coordinates for defining the domain
b_ox = 0.
b_oy = 0.
b_oz = 0.

t_ox = 0.
t_oy = 0.
t_oz = 4.58

b_ix = 0.
b_iy = 0.
b_iz = 0.

t_ix = 0.
t_iy = 0.
t_iz = 4.58

height = t_iz - b_iz #hieght of the cylinder

domain = Cylinder(Point(b_ox, b_oy, b_oz), Point(t_ox, t_oy, t_oz), outer_top_radius, outer_bot_radius) - Cylinder(Point(b_ix, b_iy, b_iz), Point(t_ix, t_iy, t_iz), inner_top_radius, inner_bot_radius)
mesh = generate_mesh ( domain, N )

test_file =File("test_plot/test_N_"+str(N)+".pvd")

if (meshrefine > .5): #technically just needs to be greater than 0 but numerical whatsit so lets be really safe
    sub_domains_bool = MeshFunction("bool",mesh,mesh.topology().dim() - 1)
    sub_domains_bool.set_all(False)

    class SmallInterest(SubDomain):
        def inside(self, x, on_boundary):
            return (x[1]**2 + x[0]**2 < (inner_top_radius+refine1)**2 or x[1]**2 + x[0]**2 > (outer_top_radius-refine1)**2)


    interest = SmallInterest()
    interest.mark(sub_domains_bool,True)
    mesh = refine(mesh,sub_domains_bool)
    print("okie dokie all good mesh is refined")
    test_file = File("test_plot/test_N_"+str(N)+"_r1_" + str(refine1)+".pvd") #This gives us a look at the mesh before we run it
    if (meshrefine > 1.5):
        class BigInterest(SubDomain):
            def inside(self, x, on_boundary):
                return (x[1]**2 + x[0]**2 < (inner_top_radius+refine2)**2 or x[1]**2 + x[0]**2 > (outer_top_radius-refine2)**2)
        #
        sub_domains_bool2 = MeshFunction("bool",mesh,mesh.topology().dim() - 1)
        sub_domains_bool2.set_all(False)
        interest = BigInterest()
        interest.mark(sub_domains_bool2,True)
        mesh = refine(mesh,sub_domains_bool2)
        print("okie dokie all good mesh is refined AGAIN wow your if statement actually worked")
        test_file = File("test_plot/test_N_"+str(N)+"_r1_" + str(refine1)+"_r2_"+str(refine2)+".pvd") #gives us a look at the mesh before we run it too long


#Moved up so
X_test = VectorFunctionSpace(mesh,"CG",2)
Q_test = FunctionSpace(mesh,"CG",1)


#Print the total number of degrees of freedom
vdof = X_test.dim()
pdof = Q_test.dim()
print("The number of velocity DOFs is:" + str(vdof))
print("The number of pressure DOFs is:" + str(pdof))

test=Function(X_test)
test_file<<test
print("Now can go look at the mesh")





#Parameters
mu = .0000055
tau = .0000001
nu = .001


Pi = np.pi
#Reynolds number info
radius_ratio = inner_bot_radius/outer_bot_radius #eta in paper
d =  outer_bot_radius - inner_bot_radius #not defined in paper
aspect_ratio =  height/d #L_x in paper
domain_volume = aspect_ratio*d*2*Pi
print(domain_volume)
Re = d*omega/nu


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
        return x[0]**2 + x[1]**2 <= inner_top_radius**2 +10**-1 and on_boundary

#Sub domain for the outer cylinder
#The tolerance had to be set much differently for this condition for some reason, double check this when swithching the mesh
class Outer_cyl(SubDomain):
    def inside(self, x, on_boundary):
        return outer_top_radius**2 - 10**-1 <= x[0]**2 + x[1]**2 and on_boundary

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




#Define boundary conditions for the velocity equation
noslip_u_inner = Expression(("mint*omega*r*x[1]/r", "mint*omega*r*-1*x[0]/r","0.0"), mint = 0.0,degree=4, r = inner_bot_radius,omega = omega)
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



eps=1.e-10
#u euqation

u_lhs = (1./dt)*inner(u,v)*dx + b(un,u,v)*dx  + 2.*nu*a_sym(u,v)*dx + eps*np.sqrt(2.)*mu*tau*kn*a_sym(u,v)*dx - c(p,v)*dx + c(q,u)*dx
u_rhs = (1./dt)*(inner(un,v))*dx

#k equation
k_lhs = (1./dt)*inner(k,phi)*dx + (nu+np.sqrt(2.)*mu*kn*tau)*a(k,phi)*dx + b(unPlus1,k,phi)*dx + (1./(np.sqrt(2.0)*tau))*inner(k,phi)*dx
k_rhs = (1./dt)*inner(kn,phi)*dx + np.sqrt(2.0)*mu*tau*kn*inner(.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T),.5*(nabla_grad(unPlus1)+nabla_grad(unPlus1).T))*phi*dx #Last term here is the square of the symmetric gradient write function for this later


Au = None
Ak = None
Bu = None
Bk = None

#Things we plot!!!
#total KE at every timestep (deceptively named? Not turbulent KE)
TKE = np.zeros((t_num))


#Location variables, probably more than we need if we're honest but I'm not taking chances
donutPoints = 100 #number of points we take along varying theta (points along donut shaped cross sections)
theta = np.zeros((donutPoints))
xy = np.zeros((2,donutPoints))
zpoints = 150 #number of points we take along z axis
avg = np.zeros((zpoints)) #average at one timestep, we save the last timestep (as god willing we've got some sort of fully evolved flow)
avgavg = np.zeros((zpoints)) #average over donut, over time, store here
z = np.zeros((zpoints))
averageCount = 0#keep track of how many things we've added to average




for i in range(0,donutPoints):
    theta[i] = i*(2*Pi)/donutPoints
    xy[0,i] = .5*(outer_top_radius+inner_top_radius)*cos(theta[i])
    xy[1,i] = .5*(outer_top_radius+inner_top_radius)*sin(theta[i])
for i in range(0,zpoints):
    z[i] = b_iz+t_iz*i/zpoints


take_snap = 10
for jj in range(0,t_num):
    t = t + dt
    print('Numerical Time Level: t = '+ str(t))

    #Matrix Assembly for the u equation
    Au = assemble(u_lhs)
    Bu = assemble(u_rhs)
    mint_val = smooth_bridge(t)
    noslip_u_inner.mint = mint_val

    #Application of boundary conditions for the u equation
    [bc.apply(Au,Bu) for bc in bcs_u]
    #Solve
    solver.solve(Au,w_.vector(),Bu)
    #Solution of the u equation
    (unPlus1,pnPlus1) = w_.split(True)
    unPlus1.vector()[:] = unPlus1.vector()[:] -(1./3.)*(unPlus1.vector()[:]-2*un.vector()[:]+unMinus1.vector()[:])
    TKE[jj]=  .5*(assemble(inner(unPlus1,unPlus1)*dx))

    #Matrix Assembly for the k equation
    Ak = assemble(k_lhs)
    Bk = assemble(k_rhs)

    #Application of boundary conditions for the k equation
    [bc.apply(Ak,Bk) for bc in bcs_k]
    #Solve
    solve(Ak,k_.vector(),Bk)
    #Solution of the u equation
    knPlus1.assign(k_)

    #Save solution, also add another round of donut averages if we've let the flow go for a bit

    if(jj%frameRate == 0):
        print(TKE[jj])
        velocity_paraview_file << (unPlus1,t)
    if(jj%take_snap == 0):
        if t>10:
            averageCount = averageCount+1
            currentavg = np.zeros((zpoints))
            for j in range(0,zpoints):
                for i in range(0,donutPoints):
                    u1 = unPlus1(xy[0,i],xy[1,i] ,z[j])[0] #velocity in x
                    u2 = unPlus1(xy[0,i],xy[1,i] ,z[j])[1] #velocity in y
                    v = u1*cos(theta[i])+u2*sin(theta[i])
                    currentavg[j]= currentavg[j] + (1./donutPoints)*v
                avgavg[j] = avgavg[j]+currentavg[j]
            np.savetxt('Avg_z/average'+str(t)+ '.txt',currentavg)
    #Assign values for next time step
    unMinus1.assign(un)
    un.assign(unPlus1)
    kn.assign(knPlus1)


for j in range(0,zpoints):
    currentavg = 0
    avgavg[j] = avgavg[j]/averageCount
    for i in range(0,donutPoints):
        u1 = unPlus1(xy[0,i],xy[1,i] ,z[j])[0] #velocity in x
        u2 = unPlus1(xy[0,i],xy[1,i] ,z[j])[1] #velocity in y
        v = u1*cos(theta[i])+u2*sin(theta[i])
        currentavg= currentavg + (1./100.)*v
    avg[j] = currentavg

np.savetxt('average_end.txt',avg)
np.savetxt('average_average.txt',avgavg)

plt.figure(1)
plt.plot(z,avg,"r", label=r"Average v at time = 11",linewidth =.5 )
plt.xlabel("z")
plt.ylabel("avg v")

plt.savefig("Avg_v_end")
plt.close()

plt.figure(2)
plt.plot(z,avgavg,"r", label=r"Average average v",linewidth =.5 )
plt.xlabel("z")
plt.ylabel("avg v")

plt.savefig("Avg_v_time_avged")
plt.close()

np.savetxt('TKE.txt',TKE)
time = np.zeros((t_num,1))
t=0
for j in range(0,t_num):
    time[j] = t
    t = t+dt

plt.figure(1)
plt.plot(time,TKE,"r", label=r"Total Kinetic Energy",linewidth =.5 )
plt.xlabel("t")
plt.ylabel("KE")

plt.savefig("Total Kinetic Energy")