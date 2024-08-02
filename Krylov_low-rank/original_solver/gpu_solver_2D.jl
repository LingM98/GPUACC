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
#using Profile
#using ProfileView

"""
Helper function to create the 2D grid
"""
function ndgrid(x::AbstractVector{T}, y::AbstractVector{T}) where {T <: Number}
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

"""
Approximately solve:
    A1 X + X A2' + U_old*S_old*V_old' = 0

Input:
    U_old, S_old, and V_old: SVD factors from the previous time level
    A1, A2: coefficient matrices
    rel_eps: Relative truncation tolerance for SVD truncation
    max_iter: Maximum number of Krylov iterations
    max_rank: Maximumm rank for SVD truncation

Output:
    U_new, S_new, V_new: Solution factors for the new time level

The iteration terminates early provided the following condition is satisfied:
    ||A1 X + X A2'- U_old*S_old*V_old'|| < ||U_old*S_old*V_old'|| * tol.
This quantity is measured by projecting onto the low-dimensional subspaces to reduce the
complexity of its formation. We use the spectral norm here.
"""
const FullOrSparseMatrix = Union{CuMatrix{Float64},CuSparseMatrixCSR{Float64, Int64}}
const FullOrDiagonalMatrix = Union{CuMatrix{Float64}, Diagonal{Float64, CuVector{Float64}}}

@fastmath @views function extended_krylov_step_gpu(U_old::CuMatrix{Float64}, V_old::CuMatrix{Float64}, S_old::FullOrDiagonalCuMatrix, 
                                  A1::FullOrSparseMatrix, A2::FullOrSparseMatrix, 
                                  rel_eps::Float64, max_iter::Int, max_rank::Int)

    # Tolerance for the construction of the Krylov basis
    threshold = CUDA.@allowscalar S[1,1]*rel_eps

    # Precompute the LU factorizations of A1 and A2
    FA1 = lu(A1)
    FA2 = lu(A2)

    # Initialize the Krylov bases
    U = copy(U_old)
    V = copy(V_old)

    # Storage for A1^{k}U and A2^{k}V for two iterates
    A1U_prev = copy(U_old)
    A2V_prev = copy(V_old)
    A1U_curr = CuMatrix{Float64}(undef, size(U_old))
    A2V_curr = CuMatrix{Float64}(undef, size(V_old))

    # Storage for A1^{-k}U and A2^{-k}V for two iterates
    inv_A1U_prev = copy(U_old)
    inv_A2V_prev = copy(V_old)
    inv_A1U_curr = CuMatrix{Float64}(undef, size(U_old))
    inv_A2V_curr = CuMatrix{Float64}(undef, size(V_old))

    # Initialize S1 here to extend its scope
    S1_d = CuMatrix{Float64}(undef, size(S_old))

    # Variables to track during construction of the Krylov subspaces
    converged = false
    num_iterations = 0

    for iter_count = 1:max_iter

        # Extended Krylov bases and concatenate with the existing bases
        mul!(A1U_curr, A1, A1U_prev)
        ldiv!(inv_A1U_curr, FA1, inv_A1U_prev)

        mul!(A2V_curr, A2, A2V_prev)
        ldiv!(inv_A2V_curr, FA2, inv_A2V_prev)

        # Orthogonalize the augmented bases
        F_U = qr!(hcat(U, A1U_curr, inv_A1U_curr))
        F_V = qr!(hcat(V, A2V_curr, inv_A2V_curr))

        U = CuMatrix(F_U.Q)
        V = CuMatrix(F_V.Q)

        # Build and solve the reduced system using the Sylvester solver
        A1U = A1*U
        A2V = A2*V

        # Compute and copy the coefficients to the CPU
        A1_tilde = Array(U'*A1U)
        A2_tilde = Array(V'*A2V)
        B1_tilde = Array((U'*U_old)*S_old*(V_old'*V))
        S1_h = sylvc(A1_tilde, A2_tilde, B1_tilde)

        # Copy S1 from the CPU back to the GPU
        S1_d = CuArray(S1_h)

        # Check convergence of the solver using the spectral norm of the residual
        # RU*[-B1_tilde S1; S1 zeros(size(S1, 1), size(S1, 2))]*RV'
        _, RU = qr!(hcat(U, A1U))
        _, RV = qr!(hcat(V, A2V))

        # Build the blocks of the matrix [-B1_tilde S1; S1 zeros(size(S1, 1), size(S1, 2))]
        residual = RU*[-B1_tilde S1_d; S1_d CUDA.zeros(size(S1_d, 1), size(S1_d, 2))]*RV'

        # Compute the spectral norm of the residual
        sigma = svdvals!(residual)
        @CUDA.allowscalar spectral_norm = sigma[1]

        if spectral_norm < threshold
            num_iterations = iter_count
            converged = true
            break
        end

        A1U_prev .= A1U_curr
        inv_A1U_prev .= inv_A1U_curr

        A2V_prev .= A2V_curr
        inv_A2V_prev .= inv_A2V_curr

    end

    # Perform SVD truncation on the dense solution tensor
    U_tilde, S_tilde, V_tilde = svd!(S1_d)

    # Here S_tilde is a vector, so we do this before
    # we promote S_tilde to a diagonal matrix
    # We can exploit the fact that S_tilde is ordered (descending)
    CUDA.@allowscalar threshold = S_tilde*rel_eps
    r = sum(S_tilde .> threshold)
    r = min(r, max_rank)

    # Define the new core tensor
    S_new = Diagonal(S_tilde[1:r])

    # Join the orthogonal bases from the QR and SVD steps
    U_new = U*U_tilde
    V_new = V*V_tilde

    return U_new, V_new, S_new, num_iterations

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

# Apply the SVD to the initial data (CPU)
Vx_old, S_old, Vy_old = svd(U_init)
S_old = Diagonal(S_old)

# Copy the SVD data to the GPU
Vx_old = CuArray(Vx_old[:, 1:2])
Vy_old = CuArray(Vy_old[:, 1:2])
S_old = CuArray(S_old[1:2,1:2])

# Call the Krylov solver
@btime begin
    Vx_new, Vy_new, S_new, iter = extended_krylov_step_cpu(Vx_old, Vy_old, S_old, A1, A2, rel_eps, max_iter, max_rank)
end

# # Use this to check for a type instability
# @code_warntype extended_krylov_step_cpu(Vx_old, Vy_old, S_old, A1, A2, rel_eps, max_iter, max_rank)

