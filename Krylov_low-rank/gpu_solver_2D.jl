# This script tests a GPU implementation of a Sylvester solver
# based on Krylov subspace methods.

using ArgParse

# Retrieve the command line arguments
settings = ArgParseSettings()

@add_arg_table settings begin
    "--Lx"
        help = "Domain length along x";
        arg_type = Float64
        default = 1.0
    "--Ly"
        help = "Domain length along y";
        arg_type = Float64
        default = 1.0
    "--Nx"
        help = "Number grid points in x";
        arg_type = Int
        default = 101
    "--Ny"
        help = "Number grid points in y";
        arg_type = Int
        default = 101
    "--rel_tol"
        help = "Relative truncation tolerance for SVD truncation"
        arg_type = Float64
        default = 1.0e-3
    "--max_rank"
        help = "Maximum rank used in the representation of the function."
        arg_type = Int
        default = 32
    "--max_iter"
        help = "Maximum number of Krylov iterations"
        arg_type = Int
        default = 10
    "--max_size"
        help = "Maximum dimension used in the preallocation phase."
        arg_type = Int
        default = 64
    "--use_mkl"
        help = "Use the Intel Math Kernel Library rather than OpenBLAS"
        arg_type = Bool
        default = false
end

# Parse the arguments and print them to the command line
parsed_args = parse_args(settings)

println("Options used:")
for (arg,val) in parsed_args
    println("  $arg  =  $val")
end

# Get the individual command line arguments from the dictionary
Lx = parsed_args["Lx"]
Ly = parsed_args["Ly"]
Nx = parsed_args["Nx"]
Ny = parsed_args["Ny"]
rel_tol = parsed_args["rel_tol"]
max_rank = parsed_args["max_rank"]
max_iter = parsed_args["max_iter"]
max_size = parsed_args["max_size"]
use_mkl = parsed_args["use_mkl"]

import Base: find_package

# Check if MKL is available. If so, use it.
if use_mkl && find_package("MKL") !== nothing
	println("Intel MKL installation found.")
	using MKL
else
	println("Running with the default OpenBLAS installation.")
end

using Printf
using BenchmarkTools
using CUDA
using LinearAlgebra
using SparseArrays
using MatrixEquations

using InteractiveUtils
# using Profile
# using ProfileView

include("SolverParameters.jl")
include("State.jl")
include("Workspace.jl")
include("extended_krylov_step.jl")

"""
Helper function to create the 2D grid
"""
function ndgrid(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Number}
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

# Get the number of BLAS threads and check the configuration
println("Number of BLAS threads:", BLAS.get_num_threads())
println("BLAS config:", BLAS.get_config())

# Setup the domain as well as the differentiation matrices
# Exclude the first and last endpoints for the boundary conditions
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
x = [i*dx for i in 0:(Nx-2)]
y = [j*dy for j in 0:(Ny-2)]
X, Y = ndgrid(x, y)

# Define most of Dxx by its diagonals.
d0 =  fill(-2/dx^2, Nx-1)    # main diagonal
dpm = ones(Nx-2)/dx^2       # super- and subdiagonal
Dxx = spdiagm(-1=>dpm, 0=>d0, 1=>dpm)
Dyy = Dxx

# Define matrices for the implicit scheme
# Can either build the matrices in a full or sparse manner
dtn = 1.0e-2           # Time step size
d1 = 0.5               # Diffusion coefficient for ddx
d2 = 0.5               # Diffusion coefficient for ddy
A1 = (1/3)*spdiagm(0 => ones(size(Dxx, 1))) - dtn*(d1^2)*Dxx
A2 = (1/3)*spdiagm(0 => ones(size(Dyy, 1))) - dtn*(d2^2)*Dyy

# Convert everything to dense arrays and copy to the device
A1 = CuArray(Matrix(A1))
A2 = CuArray(Matrix(A2))

# Create the initial data
U_init = @. 0.5 * exp(-400 * (X - 0.3)^2 - 400 * (Y - 0.35)^2 ) + 
     0.8 * exp(-400 * (X - 0.65)^2 - 400 * (Y - 0.5)^2 )

# Apply the SVD to the initial data
Vx_old, S_old, Vy_old = svd(U_init)
S_old = Diagonal(S_old)

# The initial data is rank two, but we could use information from
# the singular values
Vx_old = CuArray(Vx_old[:, 1:2])
Vy_old = CuArray(Vy_old[:, 1:2])
S_old = CuArray(S_old[1:2,1:2])

# Create the low-rank state
state_old = State2D(Vx_old, S_old, Vy_old, 2)

# Setup the parameters for the Krylov iterations
solver_params = SolverParameters(max_iter = max_iter, max_rank = max_rank, max_size = max_size, rel_tol = rel_tol, backend = CUDA_backend())

# Create the reusable workspace for the Krylov method
# This interface can probably be simplied a bit by creating an additional data structure
ws = setup_workspaces(typeof(Vx_old), A1, A2, size(Vx_old), size(Vy_old), max_size, solver_params)

# #@code_warntype extended_krylov_step!(state_old, ws, solver_params)

# Call the Sylvester solver
@btime begin
    state_new, iter = extended_krylov_step!(state_old, ws, solver_params)    
end




