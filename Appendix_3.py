from fenics import *
import numpy as np
from ufl import nabla_div
import sympy as sym

def solver(f, phi, K1_val, K2_val, K3_val, b_12_val, b_23_val, mesh, degree):
    """
    Solving the Reduced Darcy multi-compartment model for 3 equal sized
    porous media with domain of Omega = [0,1] x [0,1] using pressure boundary
    conditions and specified intercompartment coupling coefficients listed
    Beta_11 = Beta_22 = Bta_33 = Beta_21 = Beta_32 = Beta_13 = Beta_31 = 0.00,
    Beta_12 = Constant value 1.0, Beta_23 = Constant value 1.0
    """

    # Creating mesh and defining function space
    Velm = FiniteElement('P', mesh.ufl_cell(), degree)
    Welm = MixedElement([Velm, Velm, Velm])
    W = FunctionSpace(mesh, Welm)

    # Defining Pressure Boundary Conditions
    inflow = Constant(1.0201)

    def boundary_L(x, on_boundary):
        return on_boundary and near(x[0], 0)

    bc_L = DirichletBC(W.sub(0), inflow, boundary_L)

    outflow = Constant(0.7257)

    def boundary_R(x, on_boundary):
        return on_boundary and near(x[0], 1)

    bc_R = DirichletBC(W.sub(2), outflow, boundary_R)

    # Collecting boundary conditions
    bcs = [bc_L, bc_R]

    # Defining variational problem
    (p_1, p_2, p_3) = TrialFunctions(W)
    (v_1, v_2, v_3) = TestFunctions(W)
    f_1, f_2, f_3 = f
    K_1 = K1_val
    K_2 = K2_val
    K_3 = K3_val

    b_12 = Constant(b_12_val)
    b_23 = Constant(b_23_val)

    # Variational Equation
    a = inner(K_1*grad(p_1), grad(v_1))*dx + inner(K_2*grad(p_2), grad(v_2))*dx + inner(K_3*grad(p_3), grad(v_3))*dx + inner(b_12*(p_1 - p_2), v_1)*dx - inner(b_12*(p_1 - p_2), v_2)*dx + inner(b_23*(p_2 - p_3), v_2)*dx
    - inner(b_23*(p_2 - p_3), v_3)*dx

    L = inner(f_1, v_1)*dx + inner(f_2, v_2)*dx + inner(f_3, v_3)*dx

    # Computing Numerical Pressure
    p = Function(W)
    solve(a == L, p, bcs)

    return p

def run_solver():
    "Run solver to compute and post-process solution"

    mesh = UnitSquareMesh(78, 78)
    Velm = FiniteElement('P', mesh.ufl_cell(), degree=2)
    Welm = MixedElement([Velm, Velm, Velm])
    W = FunctionSpace(mesh, Welm)

    # Units
    s = 1
    mm = 1
    Pa = 1
    kg = 1000 * Pa*mm*s**2

    # Setting up problem specific parameters and calling solver
    K1_val = 0.1 * mm**2/Pa/s
    K_1 = K1_val
    K2_val = 0.5 * mm**2/Pa/s
    K_2 = K2_val
    K3_val = 0.8 * mm**2/Pa/s
    K_3 = K3_val

    phi = Constant(0.3)

    b_12_val = 1.0
    b_23_val = 1.0
    b_12 = Constant(b_12_val)
    b_23 = Constant(b_23_val)

    f_1 = 0.0 * mm**3/s
    f_2 = 0.0 * mm**3/s
    f_3 = 1e-3 * mm**3/s
    #Both f_1 and f_2 are set at zero so the compartments 1 and 2 act only as conduits and conduct flow of fluid through them without adding any pressure to the system. This way, fluid flow is dictated by pressure gradients and beta values.
    #f_3 is set at a minimal value 1e-3 so the compartment 3 acts as a sink for the fluid to pass through to its right side. Using a minimal value to keep the pressure added to the system at a minimal level so flow is dictated mainly by pressure gradient and beta values.
    f1 = Constant(f_1)
    f2 = Constant(f_2)
    f3 = Constant(f_3)

    p = solver((f1, f2, f3), phi, K_1, K_2, K_3, b_12_val, b_23_val, mesh, 2)
    # Pressure, p above is the coefficient vector of the mixed function space. To obtain components:

    p_1, p_2, p_3 = p.split(deepcopy=True)

    # Evaluating and projecting Darcy velocity profile w for each compartment
    w1_1 = -K_1*grad(p_1)
    w2_2 = -K_2*grad(p_2)
    w3_3 = -K_3*grad(p_3)
    w_1 = project(w1_1, VectorFunctionSpace(mesh, 'P', degree=2))
    w_2 = project(w2_2, VectorFunctionSpace(mesh, 'P', degree=2))
    w_3 = project(w3_3, VectorFunctionSpace(mesh, 'P', degree=2))

    # Evaluating and projecting Fluid velocity profile uf for each compartment
    uf_1_1 = w1_1/phi
    uf_1 = project(uf_1_1, VectorFunctionSpace(mesh, 'P', degree=2))
    uf_2_2 = w2_2/phi
    uf_2 = project(uf_2_2, VectorFunctionSpace(mesh, 'P', degree=2))
    uf_3_3 = w3_3/phi
    uf_3 = project(uf_3_3, VectorFunctionSpace(mesh, 'P', degree=2))

    # Saving Numerical solutions for visualization
    xdmf = XDMFFile('Pressure_Gradient_p_1.xdmf')
    p_1.rename('p1', '')
    xdmf.write(p_1)
    xdmf.close()

    xdmf = XDMFFile('Pressure_Gradient_p_2.xdmf')
    p_2.rename('p2', '')
    xdmf.write(p_2)
    xdmf.close()

    xdmf = XDMFFile('Pressure_Gradient_p_3.xdmf')
    p_3.rename('p3', '')
    xdmf.write(p_3)
    xdmf.close()

    xdmf = XDMFFile('Darcy_velocity_1.xdmf')
    w_1.rename('Darcy_velocity_1', '')
    xdmf.write(w_1)
    xdmf.close()

    xdmf = XDMFFile('Darcy_velocity_2.xdmf')
    w_2.rename('Darcy_velocity_2', '')
    xdmf.write(w_2)
    xdmf.close()

    xdmf = XDMFFile('Darcy_velocity_3.xdmf')
    w_3.rename('Darcy_velocity_3', '')
    xdmf.write(w_3)
    xdmf.close()

    xdmf = XDMFFile('Fluid_velocity_1.xdmf')
    uf_1.rename('Fluid_velocity_1', '')
    xdmf.write(uf_1)
    xdmf.close()

    xdmf = XDMFFile('Fluid_velocity_2.xdmf')
    uf_2.rename('Fluid_velocity_2', '')
    xdmf.write(uf_2)
    xdmf.close()

    xdmf = XDMFFile('Fluid_velocity_3.xdmf')
    uf_3.rename('Fluid_velocity_3', '')
    xdmf.write(uf_3)
    xdmf.close()

if __name__ == '__main__':
    run_solver()
