# This script tests a CPU implementation of a Sylvester solver
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
    "--rel_eps"
        help = "Relative truncation tolerance for SVD truncation"
        arg_type = Float64
        default = 1.0e-3
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
rel_eps = parsed_args["rel_eps"]
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
using LinearAlgebra
using SparseArrays
using MatrixEquations

include("diffmat2.jl")
include("sylvester_extended_krylov.jl")

"""
Helper function to create the 2D grid
"""
function ndgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

# Get the number of BLAS threads and check the configuration
println("Number of BLAS threads:", BLAS.get_num_threads())
println("BLAS config:", BLAS.get_config())

# Setup the domain for the problem, including the differentiation
# matrices
xspan = [0, Lx] 
yspan = [0, Ly]

x, _, Dxx = diffmat2(Nx, xspan)
y, _, Dyy = diffmat2(Ny, yspan)

# Exclude the first and last endpoints for the boundary conditions
x = x[2:end-1]
y = y[2:end-1]

Dxx = Dxx[2:end-1,2:end-1]
Dyy = Dyy[2:end-1,2:end-1]

X, Y = ndgrid(x, y)

# Create the initial data
U = 0.5 * exp.(-400 * (X .- 0.3).^2 .- 400 * (Y .- 0.35).^2 ) .+ 
     0.8 * exp.(-400 * (X .- 0.65).^2 .- 400 * (Y .- 0.5).^2 )

# Apply the SVD to the initial data
Vx_n, S_n, Vy_n = svd(U)
S_n = Diagonal(S_n)

# The initial data is rank two, but we could use information from
# the singular values
Vx_n = Vx_n[:, 1:2]
Vy_n = Vy_n[:, 1:2]
S_n = S_n[1:2,1:2]

# Define matrices for the implicit scheme
# Can either build the matrices in a full or sparse manner
dtn = 1.0e-2           # Time step size
d1 = 1/2               # Diffusion coefficient for ddx
d2 = 1/2               # Diffusion coefficient for ddy
A = (1/3) * I(size(Dxx, 1)) - dtn * d1^2 * Dxx
B = (1/3) * I(size(Dyy, 1)) - dtn * d2^2 * Dyy 
#A = (1/3) * spdiagm(0 => ones(size(Dxx, 1))) - dtn * d1^2 * sparse(Dxx)  
#B = (1/3) * spdiagm(0 => ones(size(Dyy, 1))) - dtn * d2^2 * sparse(Dyy)

# Call the Sylvester solver
Vx_nn, Vy_nn, S_nn, s = sylvester_extended_krylov(Vx_n, Vy_n, S_n, A, B, rel_eps)

# Reset defaults for the number of samples and total time for
# the benchmarking process
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

benchmark_data = @benchmark sylvester_extended_krylov(Vx_n, Vy_n, S_n, A, B, rel_eps)

# Times are in nano-seconds (ns) which are converted to seconds
sample_times = benchmark_data.times
sample_times /= 10^9

@printf "CPU results:\n"
@printf "Minimum (s): %.8e\n" minimum(sample_times)
@printf "Maximum (s): %.8e\n" maximum(sample_times)
@printf "Median (s): %.8e\n" median(sample_times)
@printf "Mean (s): %.8e\n" mean(sample_times)
@printf "Standard deviation (s): %.8e\n" std(sample_times)

