# This script tests a CPU LU decomposition of a matrix based
# on finite differences. It benchmarks the factorization A = LU, 
# where A is an N x N matrix

using ArgParse
using Printf
using BenchmarkTools
using LinearAlgebra
using SparseArrays

# Retrieve the command line arguments
s = ArgParseSettings()

@add_arg_table s begin
    "-N", "--N"
        help = "Size of the square matrix A";
        arg_type = Int
        default = 256
    "-s", "--s"
        help = "Number of samples to use for statistics"
        arg_type = Int
        default = 10
    "--use_full"
        help = "Test the LU factorization using full matrices"
        arg_type = Bool
        default = true
end

# Parse the arguments and print them to the command line
parsed_args = parse_args(s)

println("Options used for LU (CPU):")
for (arg,val) in parsed_args
    println("  $arg  =  $val")
end

# Get the individual command line arguments from the dictionary
N = parsed_args["N"]
s = parsed_args["s"]
use_full = parsed_args["use_full"]

# Get the number of BLAS threads being used
println("Number of BLAS threads:", BLAS.get_num_threads())

# Reset defaults for the number of samples and the total time for
# the benchmarking process
BenchmarkTools.DEFAULT_PARAMETERS.samples = s
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

# Setup the differentiation matrices for the operation
h = 1/(N-1)

# Define most of Dxx by its diagonals.
d0 =  fill(-2/h^2, N-1)    # main diagonal
dpm = ones(N-2)/h^2       # super- and subdiagonal
Dxx = spdiagm(-1=>dpm, 0=>d0, 1=>dpm)

# Define matrices for the implicit scheme
dtn = 1.0e-2           # Time step size
d1 = 0.5               # Diffusion coefficient for ddx
A = (1/3)*spdiagm(0 => ones(size(Dxx, 1))) - dtn*(d1^2)*Dxx

# Is A supposed to be full or sparse?
if use_full
    A = Matrix(A)
end

# Perform the LU of A
benchmark_data = @benchmark lu($A)

# Times are in nano-seconds (ns) which are converted to seconds
sample_times = benchmark_data.times
sample_times /= 10^9

@printf "CPU results:\n"
@printf "Minimum (s): %.8e\n" minimum(sample_times)
@printf "Maximum (s): %.8e\n" maximum(sample_times)
@printf "Median (s): %.8e\n" median(sample_times)
@printf "Mean (s): %.8e\n" mean(sample_times)
@printf "Standard deviation (s): %.8e\n" std(sample_times)

