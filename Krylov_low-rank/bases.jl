include("State.jl")
include("Workspace.jl")
include("SolverParameters.jl")

include("materialize_Q.jl")


"""
Function to initialize the bases in the Krylov method. This method is applied prior to the Krylov iteration.
"""
@views function initialize_bases!(ws::ExtendedKrylovWorkspace2D, state_in::State2D)

    # Set the number of columns for the basis
    ws.U_ncols = size(state_in.U,2)
    ws.V_ncols = size(state_in.V,2)

    @sync begin
        Threads.@spawn ws.U[:,1:ws.U_ncols] .= state_in.U[:,:]
        Threads.@spawn ws.V[:,1:ws.V_ncols] .= state_in.V[:,:]

        Threads.@spawn ws.A1U_prev .= state_in.U[:,:]
        Threads.@spawn ws.A2V_prev .= state_in.V[:,:]

        Threads.@spawn ws.inv_A1U_prev .= state_in.U[:,:]
        Threads.@spawn ws.inv_A2V_prev .= state_in.V[:,:]
    end

    return nothing
end

"""
Updates and orthogonalizes the bases from the Krylov iteration. Valid for any workspace.
"""
function update_bases_and_orthogonalize!(ws::AbstractWorkspace, backend::Backend)

    update_bases!(ws)
    orthogonalize_bases!(ws, backend)

    return nothing
end

"""
Helper function that calculates the candidate bases during the Krylov iteration.
"""
function update_bases!(ws::ExtendedKrylovWorkspace2D)

    # Compute the candidate bases for the Krylov interation independently
    @sync begin
        Threads.@spawn mul!(ws.A1U_curr, ws.A1, ws.A1U_prev)
        Threads.@spawn ldiv!(ws.inv_A1U_curr, ws.FA1, ws.inv_A1U_prev)
        Threads.@spawn mul!(ws.A2V_curr, ws.A2, ws.A2V_prev)
        Threads.@spawn ldiv!(ws.inv_A2V_curr, ws.FA2, ws.inv_A2V_prev)
    end

    return nothing
end

"""
Helper function that applies QR factorization to orthogonalize the bases.
"""
@views function orthogonalize_bases!(ws::ExtendedKrylovWorkspace2D, backend::Backend)

    # Augment and apply QR factorization to the bases in each dimension independently
    @sync begin
        Threads.@spawn begin
            U_aug = hcat(ws.U[:,1:ws.U_ncols], ws.A1U_curr, ws.inv_A1U_curr)
            Q_U, _ = qr!(U_aug)
            Q_U = materialize_Q(Q_U, backend)
            ws.U_ncols = size(Q_U,2)
            ws.U[:,1:ws.U_ncols] .= Q_U[:,:]
        end

        Threads.@spawn begin
            V_aug = hcat(ws.V[:,1:ws.V_ncols], ws.A2V_curr, ws.inv_A2V_curr)
            Q_V, _ = qr!(V_aug)
            Q_V = materialize_Q(Q_V, backend)
            ws.V_ncols = size(Q_V,2)
            ws.V[:,1:ws.V_ncols] .= Q_V[:,:]
        end
    end

    return nothing
end

"""
Helper function to swap the buffers for the Krylov iteration.
"""
function shuffle_iterates!(ws::ExtendedKrylovWorkspace2D)

    # Current values become previous in the next iteration
    @sync begin
        Threads.@spawn ws.A1U_prev .= ws.A1U_curr
        Threads.@spawn ws.inv_A1U_prev .= ws.inv_A1U_curr

        Threads.@spawn ws.A2V_prev .= ws.A2V_curr
        Threads.@spawn ws.inv_A2V_prev .= ws.inv_A2V_curr        
    end

    return nothing
end

"""
Performs SVD truncation of the solution. Valid for any workspace and backend.

Computes the SVD of the dense core and then joins the orthogonal bases with those
obtained from the Krylov iteration to define a new (truncated) state.
"""
@views function apply_svd_truncation!(ws::AbstractWorkspace, params::SolverParameters)
    
    # Extract some parameters from the input struct
    max_rank, rel_tol, backend = params.max_rank, params.rel_tol, params.backend
    
    # Perform SVD truncation on the dense matrix S1
    # Here we avoid the copy from host to device by reusing a
    # component from the residual evaluation.
    S1 = copy(ws.block_matrix[ws.U_ncols .+ (1:ws.U_ncols),1:ws.V_ncols])
    U_tilde, S_tilde, V_tilde = svd!(S1)

    # We exploit the fact that the vector S_tilde is ordered (descending)
    threshold = get_threshold(S_tilde, rel_tol, backend)
    r_new = sum(S_tilde .> threshold)
    r_new = min(r_new, max_rank)

    U_new = similar(ws.U, size(ws.U,1), r_new)
    S_new = Diagonal(S_tilde[1:r_new])
    V_new = similar(ws.V, size(ws.V,1), r_new)

    # Join the orthogonal bases from the QR and SVD steps
    #@sync begin
    #    CUDA.@allowscalar Threads.@spawn mul!(U_new, ws.U[:,1:ws.U_ncols], U_tilde[:,1:r_new])
    #    CUDA.@allowscalar Threads.@spawn mul!(V_new, ws.V[:,1:ws.V_ncols], V_tilde[:,1:r_new])
    #end

    @sync begin
	Threads.@spawn U_new[:,:] .= copy(ws.U[:,1:ws.U_ncols]) * copy(U_tilde[:,1:r_new])
	Threads.@spawn V_new[:,:] .= copy(ws.V[:,1:ws.V_ncols]) * copy(V_tilde[:,1:r_new])
    end

    # Create the truncated low-rank state
    state_new = State2D(U_new, S_new, V_new, r_new)

    return state_new
end

"""
Computes the threshold for the termination of the iterative scheme. The threshold is defined
according to the spectral norm of a given state. Note that we absorb the term representing the
denominator in the tolerance. 
"""
@inline function get_threshold(S::AbstractVector, rel_tol::Real, backend::CPU_backend)::Real
    return S[1]*rel_tol
end

@inline function get_threshold(S::AbstractMatrix, rel_tol::Real, backend::CPU_backend)::Real
    return S[1,1]*rel_tol
end

@static if @isdefined(CUDA)
    @inline function get_threshold(S::AbstractVector, rel_tol::Real, backend::Union{CUDA_backend, CUDA_UVM_backend})
        return CUDA.@allowscalar S[1]*rel_tol
    end

    @inline function get_threshold(S::AbstractMatrix, rel_tol::Real, backend::Union{CUDA_backend, CUDA_UVM_backend})
        return CUDA.@allowscalar S[1,1]*rel_tol
    end
end

