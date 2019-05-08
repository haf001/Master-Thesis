from fenics import *
import numpy as np
from ufl import nabla_div
import sympy as sym
import matplotlib.pyplot as plt

def solver(f, p_e, K_val, mesh, degree):
    """
    Solving the Darcy flow equation for a Unit Square Medium with Pressure Boundary Conditions.
    """

    # Creating mesh and defining function space
    V = FunctionSpace(mesh, 'P', degree)

    # Defining Dirichlet boundary
    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, p_e, boundary)

    # Defining variational problem
    p = TrialFunction(V)
    v = TestFunction(V)
    K = K_val
    a = dot(K*grad(p), grad(v))*dx
    L = inner(f, v)*dx

    # Computing Numerical Pressure
    p = Function(V)
    solve(a == L, p, bc)

    return p

def run_solver():
    "Run solver to compute and post-process solution"

    mesh = UnitSquareMesh(100, 100)

    # Setting up problem specific parameters where p_e = sin(x)*sin(y) and calling solver
    d = 2
    I = Identity(d)
    M = Expression('fmax(0.10, exp(-pow(10.0*x[1]-1.0*sin(10.0*x[0])-5.0, 2)))', degree=2, domain=mesh)
    K = M*I

    p_e = Expression('sin(x[0])*sin(x[1])', degree=2)

    # The components for term grad_p_e below is made up of the differentials of p_e with respect to x and y respectively
    grad_p_e = Expression(('cos(x[0])*sin(x[1])', 'sin(x[0])*cos(x[1])'), degree=2, domain=mesh)

    # Defining the source term
    f = nabla_div(dot(-K, grad_p_e))

    # Calling Solver
    p = solver(f, p_e, K, mesh, 1)

    # Evaluating and projecting the Velocity profile, u_bar
    u_bar1 = -K*grad(p)
    u_bar = project(u_bar1, VectorFunctionSpace(mesh, 'P', degree=1))

    # Saving and Plotting Numerical solutions for visualization
    xdmf = XDMFFile('Numerical_Pressure_Gradient_V2.xdmf')
    xdmf.write(p)
    xdmf.close()

    xdmf = XDMFFile('Velocity_Profile_V2.xdmf')
    xdmf.write(u_bar)
    xdmf.close()

def test_solver():
    "Test solver by reproducing Pressure p = sin(x)*sin(y) and generating plots"

    # Setting up parameters for testing
    p_e = Expression('sin(x[0])*sin(x[1])', degree=2)

    # Iterating over mesh number m, and appending the respective Cell Sizes h, resulting L2 Error E, Degree of freedom, DOF and integral of grad(p)*grad(p) over domain, GPS
    E = []
    h = []
    DOF = []
    GPS = []
    GPES = []

    for m in range (4, 380, 4):
        mesh = UnitSquareMesh(m, m)
        V = FunctionSpace(mesh, 'P', 1)
        p_e_f = interpolate(p_e, FunctionSpace(mesh, 'P', 2))
        d = 2
        I = Identity(d)
        M = Expression('fmax(0.10, exp(-pow(10.0*x[1]-1.0*sin(10.0*x[0])-5.0, 2)))', degree=2, domain=mesh)
        K = M*I
        grad_p_e = Expression(('cos(x[0])*sin(x[1])', 'sin(x[0])*cos(x[1])'), degree=2, domain=mesh)
        f = nabla_div(dot(-K, grad_p_e))

        # Calling solver
        p = solver(f, p_e, K, mesh, degree=1)

        # Computing for L2 Error Norm and Cell Sizes h
        E1 = errornorm(p_e_f, p, 'L2')
        print('E1=', E1)
        E.append(E1)
        h.append(mesh.hmin())
        DOF.append(len(V.dofmap().dofs()))
        IGPS = assemble(inner(grad(p), grad(p))*dx)
        GPS.append(IGPS)
        IGPES = assemble(inner(grad(p_e_f), grad(p_e_f))*dx)
        GPES.append(IGPES)

    Ea = np.array(E)
    ha = np.array(h)
    DOFa = np.array(DOF)
    GPSa = np.array(GPS)
    GPESa = np.array(GPES)

    # Computing the Logs of L2 Error and Cell Sizes h with print of the Convergence Rate
    LogEa = np.log(Ea)
    Logha = np.log(ha)
    LogDOFa = np.log(DOFa)
    LogGPSa = np.log(GPSa)
    print(np.polyfit(Logha, LogEa, deg=1))

    return (E, h, DOF, GPS, GPES)

if __name__ == '__main__':
    run_solver()

    E, h, DOF, GPS, GPES = test_solver()

    # Log plot of L2 Error E against Cell Sizes h, the gradient of the line produces the Convergence Rate as well
    x = np.log(h)
    y = np.log(E)
    plt.plot(x,y)
    plt.title('Log of L2 Error vs. Log of Cell Sizes h')
    plt.xlabel('Log Cell Size, h')
    plt.ylabel('Log L2 Error')
    plt.savefig('Log_L2_Error_vs_Log_Cell_Sizes_h.png')

    # Semilog plot of L2 Error against DOF
    plt.figure()
    plt.semilogy(DOF, E)
    plt.title('Semilog L2 Error vs DOF')
    plt.xlabel('DOF')
    plt.ylabel('L2 Error')
    plt.savefig('Semilog_L2_Error_vs_DOF.png')

    # Log plot of L2 Error against DOF
    plt.figure()
    x = np.log(DOF)
    y = np.log(E)
    plt.plot(x,y)
    plt.title('Log L2 Error vs. Log DOF')
    plt.xlabel('Log DOF')
    plt.ylabel('Log L2 Error')
    plt.savefig('Log_L2_Error_vs_Log_DOF.png')

    # Log plot of Integral of grad(p) squared over Omega against Cell Sizes h for Mesh Independence
    plt.figure()
    x = np.log(h)
    y = np.log(GPS)
    plt.plot(-x,y)
    plt.axvline(x=4.0, color='r', linestyle='dotted')
    plt.axhline(y=-0.92468, color='r', linestyle='dotted')
    plt.title('Log Integral of grad(p)*grad(p) over omega vs. -Log Cell Size, h')
    plt.xlabel('-Log Cell Size, h')
    plt.ylabel('Log of Integral of grad(p)*grad(p) over omega')
    plt.savefig('Log_integral_of_grad_p_square_vs_-Log_Cell_Size_h.png')

    # Log plot of (GPES-GPS) against Cell Size, h
    plt.figure()
    x = h
    y = np.log(GPS)-np.log(GPES)
    plt.plot(-np.log(x), y)
    plt.axvline(x=4.0, color='r', linestyle='dotted')
    plt.axhline(y=0.00001, color='r', linestyle='dotted')
    plt.title('Log GPES - Log GPS vs. -Log Cell Size, h')
    plt.xlabel('-Log Cell Size, h')
    plt.ylabel('Log GPES - Log GPS')
    plt.savefig('Log_GPES-Log_GPS_vs_-Log_Cell_Size_h.png')
