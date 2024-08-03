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
        default = 1024
    "--Ny"
        help = "Number grid points in y";
        arg_type = Int
        default = 1024
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
using CUDA, CUDA.CUSPARSE
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
const FullOrSparseCuMatrix = Union{CuMatrix{Float64},CuSparseMatrixCSR{Float64, Int32}}
const FullOrDiagonalCuMatrix = Union{CuMatrix{Float64}, Diagonal{Float64, CuArray{Float64, 1, CUDA.DeviceMemory}}}

@fastmath @views function extended_krylov_step_gpu(U_old::CuMatrix{Float64}, V_old::CuMatrix{Float64}, S_old::FullOrDiagonalCuMatrix, 
                                  A1::FullOrSparseCuMatrix, A2::FullOrSparseCuMatrix, 
                                  rel_tol::Float64, max_iter::Int, max_rank::Int)

    # Tolerance for the construction of the Krylov basis
    threshold = CUDA.@allowscalar S_old[1,1]*rel_tol

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

    # Define S1 here to extend its scope
    S1_h = Matrix{Float64}(undef, size(S_old))
    S1_d = CuMatrix{Float64}(undef, size(S_old))

    # Create a container to hold references to the two R matrices defined in the tasks
    R_container = Vector{CuMatrix{Float64}}(undef, 2)

    # Create a container to hold the references to the sylvester coefficients defined in the tasks
    # These are present on the host
    sylvester_coeffs = Vector{Matrix{Float64}}(undef, 3)

    # Variables to track during construction of the Krylov subspaces
    converged = false
    num_iterations = 0

    for iter_count = 1:max_iter

        @sync begin
            # Extend the Krylov bases
            A1U_basis = Threads.@spawn mul!(A1U_curr, A1, A1U_prev)
            A2V_basis = Threads.@spawn mul!(A2V_curr, A2, A2V_prev)
            inv_A1U_basis = Threads.@spawn ldiv!(inv_A1U_curr, FA1, inv_A1U_prev)
            inv_A2V_basis = Threads.@spawn ldiv!(inv_A2V_curr, FA2, inv_A2V_prev)

            # Orthogonalize the augmented bases
            U_orthog = Threads.@spawn begin
                wait(A1U_basis), wait(inv_A1U_basis)
                F_U = qr!(hcat(U, A1U_curr, inv_A1U_curr))
                U = CuMatrix(F_U.Q)
            end

            V_orthog = Threads.@spawn begin
                wait(A2V_basis), wait(inv_A2V_basis)
                F_V = qr!(hcat(V, A2V_curr, inv_A2V_curr))
                V = CuMatrix(F_V.Q)
            end

            # Compute the coefficients for the Sylvester equation (DtH)
            A1_tilde_DtH = Threads.@spawn begin
                wait(U_orthog)
                A1_tilde = Array(U'*A1*U)
		sylvester_coeffs[1] = A1_tilde
            end

            A2_tilde_DtH = Threads.@spawn begin
                wait(V_orthog)
                A2_tilde = Array(V'*A2*V)
		sylvester_coeffs[2] = A2_tilde
            end

            B1_tilde_DtH = Threads.@spawn begin
                wait(U_orthog), wait(V_orthog)
                B1_tilde = Array((U'*U_old)*S_old*(V_old'*V))
		sylvester_coeffs[3] = B1_tilde
            end

            sylvester_solve_HtD = Threads.@spawn begin
                # Wait until the coefficients are available
                wait(A1_tilde_DtH), wait(A2_tilde_DtH), wait(B1_tilde_DtH)

		A1_tilde_h = sylvester_coeffs[1]
		A2_tilde_h = sylvester_coeffs[2]
		B1_tilde_h = sylvester_coeffs[3]

                # Solve the Sylvester equation on the CPU
                S1_h = sylvc(A1_tilde_h, A2_tilde_h, B1_tilde_h)

                # Copy the data back to GPU memory
                S1_d = CuArray(S1_h)
            end

            # Compute the R matrices for the evaluation of the residual
            RU_residual = Threads.@spawn begin
		wait(U_orthog)
                A1U = A1*U
                _, RU = qr!(hcat(U, A1U))
                R_container[1] = RU
            end

            RV_residual = Threads.@spawn begin
		wait(V_orthog)
                A2V = A2*V
                _, RV = qr!(hcat(V, A2V))
                R_container[2] = RV
            end           

            # Check the convergence of the solver using the spectral norm of the residual
            # We recompute B1_tilde to avoid communications
            Threads.@spawn begin

                wait(U_orthog), wait(V_orthog)
                B1_tilde = (U'*U_old)*S_old*(V_old'*V)

                wait(sylvester_solve_HtD)
                block_matrix = [-B1_tilde S1_d; S1_d CUDA.zeros(size(S1_d))]

                wait(RU_residual), wait(RV_residual)
                RU = R_container[1]
		RV = R_container[2]

		#@show size(RU)
		#@show size(block_matrix)
		#@show size(RV)

		residual = RU*block_matrix*RV'
        
                sigma = svdvals!(residual)
                @CUDA.allowscalar converged = sigma[1] < threshold  
            end
        end

        if converged
            num_iterations = iter_count
            break
        end

        # Swap iterates for the next pass
        @sync begin
            Threads.@spawn A1U_prev .= A1U_curr
            Threads.@spawn inv_A1U_prev .= inv_A1U_curr
            Threads.@spawn A2V_prev .= A2V_curr
            Threads.@spawn inv_A2V_prev .= inv_A2V_curr 
        end
    end

    # Perform SVD truncation on the dense solution tensor
    U_tilde, S_tilde, V_tilde = svd!(S1_d)

    # Here S_tilde is a vector, so we do this before
    # we promote S_tilde to a diagonal matrix
    # We can exploit the fact that S_tilde is ordered (descending)
    CUDA.@allowscalar threshold = S_tilde[1]*rel_tol
    r = sum(S_tilde .> threshold)
    r = min(r, max_rank)

    # Define the new core tensor and join the bases
    S_new = Diagonal(S_tilde[1:r])
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

CUDA.synchronize()

#@printf "Preparing to start profiling...\n"

#CUDA.@profile begin
#@btime begin
#    extended_krylov_step_gpu(Vx_old, Vy_old, S_old, A1, A2, rel_tol, max_iter, max_rank)
#    synchronize()
#end

BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

# Call the Krylov solver
benchmark_data = @benchmark CUDA.@sync extended_krylov_step_gpu(Vx_old, Vy_old, S_old, A1, A2, rel_tol, max_iter, max_rank)

# Times are in nano-seconds (ns) which are converted to seconds
sample_times = benchmark_data.times
sample_times /= 10^9

@printf "GPU results:\n"
@printf "Minimum (s): %.8e\n" minimum(sample_times)
@printf "Maximum (s): %.8e\n" maximum(sample_times)
@printf "Median (s): %.8e\n" median(sample_times)
@printf "Mean (s): %.8e\n" mean(sample_times)
@printf "Standard deviation (s): %.8e\n" std(sample_times)




# Use this to check for a type instability
#@code_warntype extended_krylov_step_gpu(Vx_old, Vy_old, S_old, A1, A2, rel_tol, max_iter, max_rank)






# # Old code....
# #
# #
# @fastmath @views function extended_krylov_step_gpu(U_old::CuMatrix{Float64}, V_old::CuMatrix{Float64}, S_old::FullOrDiagonalCuMatrix, 
#     A1::FullOrSparseCuMatrix, A2::FullOrSparseCuMatrix, 
#     rel_tol::Float64, max_iter::Int32, max_rank::Int32)

# # Tolerance for the construction of the Krylov basis
# threshold = CUDA.@allowscalar S_old[1,1]*rel_tol

# # Precompute the LU factorizations of A1 and A2
# FA1 = lu(A1)
# FA2 = lu(A2)

# # Initialize the Krylov bases
# U = copy(U_old)
# V = copy(V_old)

# # Storage for A1^{k}U and A2^{k}V for two iterates
# A1U_prev = copy(U_old)
# A2V_prev = copy(V_old)
# A1U_curr = CuMatrix{Float64}(undef, size(U_old))
# A2V_curr = CuMatrix{Float64}(undef, size(V_old))

# # Storage for A1^{-k}U and A2^{-k}V for two iterates
# inv_A1U_prev = copy(U_old)
# inv_A2V_prev = copy(V_old)
# inv_A1U_curr = CuMatrix{Float64}(undef, size(U_old))
# inv_A2V_curr = CuMatrix{Float64}(undef, size(V_old))

# # Define S1 here to extend its scope
# S1_d = CuMatrix{Float64}(undef, size(S_old))

# # Variables to track during construction of the Krylov subspaces
# converged = false
# num_iterations = 0

# for iter_count = 1:max_iter

# # Extended Krylov bases and concatenate with the existing bases
# mul!(A1U_curr, A1, A1U_prev)
# ldiv!(inv_A1U_curr, FA1, inv_A1U_prev)

# mul!(A2V_curr, A2, A2V_prev)
# ldiv!(inv_A2V_curr, FA2, inv_A2V_prev)

# # Orthogonalize the augmented bases
# F_U = qr!(hcat(U, A1U_curr, inv_A1U_curr))
# F_V = qr!(hcat(V, A2V_curr, inv_A2V_curr))

# U = CuMatrix(F_U.Q)
# V = CuMatrix(F_V.Q)

# # Build and solve the reduced system using the Sylvester solver
# A1U = A1*U
# A2V = A2*V

# # Compute and copy the coefficients to the CPU
# A1_tilde = Array(U'*A1U)
# A2_tilde = Array(V'*A2V)
# B1_tilde = Array((U'*U_old)*S_old*(V_old'*V))
# S1_h = sylvc(A1_tilde, A2_tilde, B1_tilde)

# # Copy S1 from the CPU back to the GPU
# S1_d = CuArray(S1_h)

# # Check convergence of the solver using the spectral norm of the residual
# # RU*[-B1_tilde S1; S1 zeros(size(S1, 1), size(S1, 2))]*RV'
# _, RU = qr!(hcat(U, A1U))
# _, RV = qr!(hcat(V, A2V))

# # Build the blocks of the matrix [-B1_tilde S1; S1 zeros(size(S1, 1), size(S1, 2))]
# block_matrix = [-CuArray(B1_tilde) S1_d; S1_d CUDA.zeros(size(S1_d))]
# residual = RU*block_matrix*RV'

# # Compute the spectral norm of the residual
# sigma = svdvals!(residual)
# @CUDA.allowscalar spectral_norm = sigma[1]

# if spectral_norm < threshold
# num_iterations = iter_count
# converged = true
# break
# end

# A1U_prev .= A1U_curr
# inv_A1U_prev .= inv_A1U_curr

# A2V_prev .= A2V_curr
# inv_A2V_prev .= inv_A2V_curr

# end

# # Perform SVD truncation on the dense solution tensor
# U_tilde, S_tilde, V_tilde = svd!(S1_d)

# # Here S_tilde is a vector, so we do this before
# # we promote S_tilde to a diagonal matrix
# # We can exploit the fact that S_tilde is ordered (descending)
# CUDA.@allowscalar threshold = S_tilde[1]*rel_tol
# r = sum(S_tilde .> threshold)
# r = min(r, max_rank)

# # Define the new core tensor
# S_new = Diagonal(S_tilde[1:r])

# # Join the orthogonal bases from the QR and SVD steps
# U_new = U*U_tilde
# V_new = V*V_tilde

# return U_new, V_new, S_new, num_iterations

# end
