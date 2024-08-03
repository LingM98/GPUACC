# Create some backend types for the different initializations
abstract type Backend end

struct CPU_backend <: Backend end
struct CUDA_backend <: Backend end
struct CUDA_UVM_backend <: Backend end

@kwdef struct SolverParameters{T1 <: Real, T2 <: Backend}
    max_iter::Int
    max_rank::Int
    max_size::Int
    rel_tol::T1
    backend::T2
end