
include("SolverParameters.jl")

# The factorizations in the extended Krylov method have a special type in the sparse case
@static if @isdefined(CUDSS)
    const Solver_t = CudssSolver{T} where T
else
    const Solver_t = Nothing
end

abstract type AbstractWorkspace end

mutable struct ExtendedKrylovWorkspace2D{T1, T2, T3} <: AbstractWorkspace
    # Coefficient matrices for the problem
    A1::T1
    A2::T1

    # Storage for the factorizations of the coefficient matrices
    FA1::Union{Factorization, Solver_t, Nothing}
    FA2::Union{Factorization, Solver_t, Nothing}

    # Container to hold the extended Krylov bases and tracker for the number of columns
    U::T2
    V::T2
    U_ncols::Int
    V_ncols::Int

    # Storage for A1^{k}U and A2^{k}V for two iterates
    A1U_prev::T2
    A2V_prev::T2
    A1U_curr::T2
    A2V_curr::T2

    # Storage for A1^{-k}U and A2^{-k}V for two iterates
    inv_A1U_prev::T2
    inv_A2V_prev::T2
    inv_A1U_curr::T2
    inv_A2V_curr::T2

    # Components for the residual measurement
    A1U::T2
    A2V::T2
    block_matrix::T2
    residual::T2

    # Components for the reduced Sylvester equation
    S1::T3
    A1_tilde::T3
    A2_tilde::T3
    B1_tilde::T3

    # Custom constructor
    function ExtendedKrylovWorkspace2D(A1::T1, A2::T1, 
        U::T2, V::T2, U_ncols::Int, V_ncols::Int, 
        A1U_prev::T2, A2V_prev::T2, A1U_curr::T2, A2V_curr::T2, 
        inv_A1U_prev::T2, inv_A2V_prev::T2, inv_A1U_curr::T2, inv_A2V_curr::T2,
        A1U::T2, A2V::T2, block_matrix::T2, residual::T2, 
        S1::T3, A1_tilde::T3, A2_tilde::T3, B1_tilde::T3) where {T1, T2, T3}

        # Create an instance with FA1 and FA2 initialized to nothing
        new{T1,T2,T3}(A1, A2, nothing, nothing, 
                      U, V, U_ncols, V_ncols, 
                      A1U_prev, A2V_prev, A1U_curr, A2V_curr, 
                      inv_A1U_prev, inv_A2V_prev, inv_A1U_curr, inv_A2V_curr,
                      A1U, A2V, block_matrix, residual, 
                      S1, A1_tilde, A2_tilde, B1_tilde)
    end
end


"""
Function to setup the workspace for the Krylov solver. 
    
Note: This includes the workspaces for the bases, the sylvester_solve, and the residual.
"""
function setup_workspaces(::Type{T}, A1::AbstractMatrix, A2::AbstractMatrix, U_size::Tuple{Int, Int}, V_size::Tuple{Int, Int}, max_size::Int, params::SolverParameters) where {T}

    # Set up the workspace for the bases of the extended Krylov method
    U, V, A1U_prev, A2V_prev, A1U_curr, A2V_curr, inv_A1U_prev, inv_A2V_prev, inv_A1U_curr, inv_A2V_curr = setup_workspace_bases(T, U_size, V_size, max_size)

    # Set up the workspace for the Sylvester equation
    S1, A1_tilde, A2_tilde, B1_tilde = setup_workspace_sylvester(params.backend, max_size)

    # Set up the workspace for the residual
    A1U, A2V, block_matrix, residual = setup_workspace_residual(T, U_size, V_size, max_size)

    # Set the current number of columns in the bases
    U_ncols = U_size[2]
    V_ncols = V_size[2]

    # Setup the workspace for the problem with empty coefficient factorizations
    ws = ExtendedKrylovWorkspace2D(A1, A2, U, V, U_ncols, V_ncols, 
                                   A1U_prev, A2V_prev, A1U_curr, A2V_curr, 
                                   inv_A1U_prev, inv_A2V_prev, inv_A1U_curr, inv_A2V_curr,
                                   A1U, A2V, block_matrix, residual, 
                                   S1, A1_tilde, A2_tilde, B1_tilde)

    return ws
end


"""
Helper function to setup the workspace for the Krylov bases. 
    
Note: This function is only responsible for preallocating memory. Additionally, the
memory will be allocated in the same location as the input arrays.
"""
function setup_workspace_bases(::Type{T}, U_size::Tuple{Int, Int}, V_size::Tuple{Int, Int}, max_size::Int) where {T}

    U_nrows, U_ncols = U_size
    V_nrows, V_ncols = V_size

    # Storage for the Krylov bases
    U = T(undef, U_nrows, max_size)
    V = T(undef, V_nrows, max_size)
    
    # Storage for A1^{k}U and A2^{k}V for two iterates
    A1U_prev = T(undef, U_nrows, U_ncols)
    A2V_prev = T(undef, V_nrows, V_ncols)
    A1U_curr = T(undef, U_nrows, U_ncols)
    A2V_curr = T(undef, V_nrows, V_ncols)
    
    # Storage for A1^{-k}U and A2^{-k}V for two iterates
    inv_A1U_prev = T(undef, U_nrows, U_ncols)
    inv_A2V_prev = T(undef, V_nrows, V_ncols)
    inv_A1U_curr = T(undef, U_nrows, U_ncols)
    inv_A2V_curr = T(undef, V_nrows, V_ncols)

    return U, V, A1U_prev, A2V_prev, A1U_curr, A2V_curr, inv_A1U_prev, inv_A2V_prev, inv_A1U_curr, inv_A2V_curr

end


"""
Helper functions to setup the workspace for the sylvester equation.
"""
function setup_workspace_sylvester(backend::CPU_backend, max_size::Int; elem_type::Type = Float64)

    S1, A1_tilde, A2_tilde, B1_tilde = create_sylvester_CPU_arrays(elem_type::Type, max_size::Int)    

    return S1, A1_tilde, A2_tilde, B1_tilde

end


@static if @isdefined(CUDA)

    function setup_workspace_sylvester(backend::CUDA_backend, max_size::Int, elem_type::Type = Float64)

        # First create the arrays on the CPU
        S1, A1_tilde, A2_tilde, B1_tilde = create_sylvester_CPU_arrays(elem_type::Type, max_size::Int) 

        # Pin the CPU memory for these arrays to reduce the transfer time
        # This also allows the scheduler to perform the copies asynchronously
        S1 = CUDA.pin(S1)
        A1_tilde = CUDA.pin(A1_tilde)
        A2_tilde = CUDA.pin(A2_tilde)
        B1_tilde = CUDA.pin(B1_tilde)

        return S1, A1_tilde, A2_tilde, B1_tilde

    end


    function setup_workspace_sylvester(backend::CUDA_UVM_backend, max_size::Int, elem_type::Type = Float64)

        # First create the arrays on the CPU
        S1, A1_tilde, A2_tilde, B1_tilde = create_sylvester_CPU_arrays(elem_type::Type, max_size::Int) 

        # Mark the above arrays so they can be accessed from any device
        S1 = cu(S1; unified=true)
        A1_tilde = cu(A1_tilde; unified = true)
        A2_tilde = cu(A2_tilde; unified = true)
        B1_tilde = cu(B1_tilde; unified = true)

        return S1, A1_tilde, A2_tilde, B1_tilde

    end

end

function create_sylvester_CPU_arrays(elem_type::T, max_size::Int) where {T}
    
    S1 = Matrix{elem_type}(undef, max_size, max_size)
    A1_tilde = Matrix{elem_type}(undef, max_size, max_size)
    A2_tilde = Matrix{elem_type}(undef, max_size, max_size)
    B1_tilde = Matrix{elem_type}(undef, max_size, max_size) 
    
    return S1, A1_tilde, A2_tilde, B1_tilde

end

"""
Helper function to setup the workspace for the residual
"""
function setup_workspace_residual(matrix_type::Type{T}, U_size::Tuple{Int, Int}, V_size::Tuple{Int, Int}, max_size::Int) where {T}

    U_nrows, _ = U_size
    V_nrows, _ = V_size

    A1U = matrix_type(undef, U_nrows, max_size)
    A2V = matrix_type(undef, V_nrows, max_size)
    
    block_matrix = matrix_type(undef, max_size, max_size)
    residual = matrix_type(undef, max_size, max_size)

    return A1U, A2V, block_matrix, residual

end

"""
Functions to initialize the LU factorizations for the coefficient matrices A1 and A2. These calls dispatch to
a method suitable for the given backend and structure of A1 and A2. These are only necessary for the extended
Krylov methods.
"""
function compute_LU_factorizations!(ws::ExtendedKrylovWorkspace2D)

    @sync begin
        Threads.@spawn ws.FA1 = lu(ws.A1) 
        Threads.@spawn ws.FA2 = lu(ws.A2)
    end

    return nothing
end
