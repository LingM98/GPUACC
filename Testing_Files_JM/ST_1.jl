# This script tests and compare the @async and @spawn macro in CUDA.
# It benchmarks @async copyto!(A2, A1) and Threads.@spawn copyto!(A2, A1), where A1, A2 are both M x N matrices

using CUDA
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

println("Options used for BandSpeedTest:")
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

T = Float64

const A_d = CUDA.randn(T, (M, N))
const B_d = CUDA.randn(T, (M, N))
const C_d = CUDA.randn(T, (M, N))

# initialize empty host array
A_h = Matrix{T}(undef, size(A_d))
B_h = Matrix{T}(undef, size(B_d))
C_h = Matrix{T}(undef, size(C_d))

# Perform the Matrix Multiplication AxA'
benchmark_data_1 = @benchmark @sync begin
    @async copyto!(A_h, A_d)
    @async copyto!(B_h, B_d)
    @async copyto!(C_h, C_d)
end

benchmark_data_2 = @benchmark @sync begin
    Threads.@spawn copyto!(A_h, A_d)
    Threads.@spawn copyto!(B_h, B_d)
    Threads.@spawn copyto!(C_h, C_d)
end

# Times are in nano-seconds (ns) which are converted to seconds
sample_times_1 = benchmark_data_1.times
sample_times_1 /= 10^9

sample_times_2 = benchmark_data_2.times
sample_times_2 /= 10^9

@printf "@async results:\n"
@printf "@spawn results:\n"

@printf "Minimum (s): %.8e\n" minimum(sample_times_1)
@printf "Minimum (s): %.8e\n" minimum(sample_times_2)

@printf "Maximum (s): %.8e\n" maximum(sample_times_1)
@printf "Maximum (s): %.8e\n" maximum(sample_times_2)

@printf "Median (s): %.8e\n" median(sample_times_1)
@printf "Median (s): %.8e\n" median(sample_times_2)

@printf "Mean (s): %.8e\n" mean(sample_times_1)
@printf "Mean (s): %.8e\n" mean(sample_times_2)

@printf "Standard deviation (s): %.8e\n" std(sample_times_1)
@printf "Standard deviation (s): %.8e\n" std(sample_times_2)
