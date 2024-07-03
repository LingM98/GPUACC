# This script tests a GPU SVD factorization.
# It benchmarks the factorization A = USV', where A is an M x N matrix

using ArgParse
using Printf
using BenchmarkTools
using CUDA

# Retrieve the command line arguments
s = ArgParseSettings()

@add_arg_table s begin
    "-M", "--M"
        help = "Number of rows in A";
        arg_type = Int
        default = 256
    "-N", "--N"
        help = "Number of columns in A";
        arg_type = Int
        default = 256
end

# Parse the arguments and print them to the command line
parsed_args = parse_args(s)

println("Options used for SVD (GPU):")
for (arg,val) in parsed_args
    println("  $arg  =  $val")
end

# Get the individual command line arguments from the dictionary
M = parsed_args["M"]
N = parsed_args["N"]

# Setup the matrices for the operation
# We declare these as "cost" so they are not treated as globals
# Alternatively, we could have used interpolation here.
const A = CUDA.randn(Float64, (M, N))

# Perform the SVD factorization A = USV'
CUDA.@profile CUDA.@sync CUDA.svd(A)


