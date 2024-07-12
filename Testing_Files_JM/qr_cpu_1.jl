# This script tests a CPU QR factorization.
# It benchmarks the factorization A = USV', where A is an M x N matrix

using ArgParse
using Printf
using BenchmarkTools
using LinearAlgebra

# Retrieve the command line arguments
s = ArgParseSettings()

@add_arg_table s begin
    "-M", "--M"
        help = "Number of rows in A";
        arg_type = Int
        default = 256
    "-N", "--N"
        help = "Number of columns in B";
        arg_type = Int
        default = 256
    "-s", "--s"
        help = "Number of samples to use for statistics"
        arg_type = Int
        default = 10
end

# Parse the arguments and print them to the command line
parsed_args = parse_args(s)

println("Options used for SVD (CPU):")
for (arg,val) in parsed_args
    println("  $arg  =  $val")
end

# Get the individual command line arguments from the dictionary
M = parsed_args["M"]
N = parsed_args["N"]
s = parsed_args["s"]

# Get the number of BLAS threads being used
println("Number of BLAS threads:", BLAS.get_num_threads())

# Reset defaults for the number of samples and the total time for
# the benchmarking process
BenchmarkTools.DEFAULT_PARAMETERS.samples = s
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1000

# Setup the matrices for the operation
# We declare these as "cost" so they are not treated as globals
# Alternatively, we could have used interpolation here.
const A = randn(Float64, (M, N))

# Perform the QR factorization A = USV'
benchmark_data = @benchmark qr(A)

# Times are in nano-seconds (ns) which are converted to seconds
sample_times = benchmark_data.times
sample_times /= 10^9

@printf "CPU results:\n"
@printf "Minimum (s): %.8e\n" minimum(sample_times)
@printf "Maximum (s): %.8e\n" maximum(sample_times)
@printf "Median (s): %.8e\n" median(sample_times)
@printf "Mean (s): %.8e\n" mean(sample_times)
@printf "Standard deviation (s): %.8e\n" std(sample_times)
