# This script tests a GPU eigenvalue decomposition.
# It benchmarks the factorization A = Q \Gamma Q', where A is an N x N matrix

using ArgParse
using Printf
using BenchmarkTools
using CUDA
using LinearAlgebra

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
    "--symmetric"
        help = "Test the eigendecomposition using symmetric matrices"
        arg_type = Bool
        default = false
end

# Parse the arguments and print them to the command line
parsed_args = parse_args(s)

println("Options used for eigen (GPU):")
for (arg,val) in parsed_args
    println("  $arg  =  $val")
end

# Get the individual command line arguments from the dictionary
N = parsed_args["N"]
s = parsed_args["s"]
make_symmetric = parsed_args["symmetric"]

# Reset defaults for the number of samples and the total time
# spent for the benchmarking process
BenchmarkTools.DEFAULT_PARAMETERS.samples = s
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 120

# Setup the matrices for the operation
# We declare these as "cost" so they are not treated as globals
# Alternatively, we could have used interpolation here.
A = CUDA.randn(Float64, (N, N))

# Is A supposed to be symmetric? If so make it symmetric and tag it
if make_symmetric
    A = A + A'
    A = Symmetric(A)
end

# Perform the SVD factorization A = USV'
benchmark_data = @benchmark CUDA.@sync eigen($A)

# Times are in nano-seconds (ns) which are converted to seconds
sample_times = benchmark_data.times
sample_times /= 10^9

@printf "GPU results:\n"
@printf "Minimum (s): %.8e\n" minimum(sample_times)
@printf "Maximum (s): %.8e\n" maximum(sample_times)
@printf "Median (s): %.8e\n" median(sample_times)
@printf "Mean (s): %.8e\n" mean(sample_times)
@printf "Standard deviation (s): %.8e\n" std(sample_times)

