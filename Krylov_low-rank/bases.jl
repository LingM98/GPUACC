include("State.jl")
include("Workspace.jl")
include("SolverParameters.jl")

# TO-DO: These functions should be written using tasks

"""
Function to initialize the bases in the Krylov method. This method is applied prior to the Krylov iteration.
"""
@fastmath @views function initialize_bases!(ws::ExtendedKrylovWorkspace2D, state_in::State2D)

    # Set the number of columns for the basis
    ws.U_ncols = size(state_in.U,2)
    ws.V_ncols = size(state_in.V,2)

    ws.U[:,1:ws.U_ncols] .= state_in.U[:,:]
    ws.V[:,1:ws.V_ncols] .= state_in.V[:,:]

    ws.A1U_prev .= state_in.U[:,:]
    ws.A2V_prev .= state_in.V[:,:]

    ws.inv_A1U_prev .= state_in.U[:,:]
    ws.inv_A2V_prev .= state_in.V[:,:]

    return nothing
end

"""
Updates and orthogonalizes the bases from the Krylov iteration. Valid for any workspace.
"""
@fastmath @views function update_bases_and_orthogonalize!(ws::AbstractWorkspace)

    update_bases!(ws)
    orthogonalize_bases!(ws)

    return nothing
end

"""
Helper function that calculates the candidate bases during the Krylov iteration.
"""
@fastmath @views function update_bases!(ws::ExtendedKrylovWorkspace2D)

    mul!(ws.A1U_curr, ws.A1, ws.A1U_prev)
    ldiv!(ws.inv_A1U_curr, ws.FA1, ws.inv_A1U_prev)

    mul!(ws.A2V_curr, ws.A2, ws.A2V_prev)
    ldiv!(ws.inv_A2V_curr, ws.FA2, ws.inv_A2V_prev)

    return nothing
end

"""
Helper function that applies QR factorization to orthogonalize the bases.
"""
@fastmath @views function orthogonalize_bases!(ws::ExtendedKrylovWorkspace2D)

    U_aug = hcat(ws.U[:,1:ws.U_ncols], ws.A1U_curr, ws.inv_A1U_curr)
    V_aug = hcat(ws.V[:,1:ws.V_ncols], ws.A2V_curr, ws.inv_A2V_curr)

    # Orthogonalize the augmented bases
    # Note that we don't explicitly need to cast to the appropriate type
    # when we transfer the bases
    F_U = qr!(U_aug)
    F_V = qr!(V_aug)

    # Update the current sizes of the bases and transfer accordingly
    ws.U_ncols = size(F_U,2)
    ws.V_ncols = size(F_V,2)

    ws.U[:,1:ws.U_ncols] .= F_U.Q[:,1:ws.U_ncols]
    ws.V[:,1:ws.V_ncols] .= F_V.Q[:,1:ws.V_ncols]

    return nothing
end

"""
Helper function to swap the buffers for the Krylov iteration.
"""
@fastmath @views function shuffle_iterates!(ws::ExtendedKrylovWorkspace2D)

    # Current values become previous in the next iteration
    ws.A1U_prev .= ws.A1U_curr
    ws.inv_A1U_prev .= ws.inv_A1U_curr

    ws.A2V_prev .= ws.A2V_curr
    ws.inv_A2V_prev .= ws.inv_A2V_curr

    return nothing
end

"""
Performs SVD truncation of the solution. Valid for any workspace and backend.

Computes the SVD of the dense core and then joins the orthogonal bases with those
obtained from the Krylov iteration to define a new (truncated) state.
"""
@fastmath @views function apply_svd_truncation!(ws::AbstractWorkspace, params::SolverParameters)
    
    # Perform SVD truncation on the dense "core"
    U_tilde, S_tilde, V_tilde = svd!(ws.S1[1:ws.U_ncols,1:ws.V_ncols])

    # Here S_tilde is a vector, so we do this before
    # we promote S_tilde to a diagonal matrix
    # We can exploit the fact that S_tilde is ordered (descending)
    r_new = sum(S_tilde .> params.rel_tol*S_tilde[1])
    r_new = min(r_new, params.max_rank)

    # Define the new diagonal "core" tensor
    # Does this work for CuArrays as well???
    S_new = Diagonal(S_tilde[1:r_new])

    # Join the orthogonal bases from the QR and SVD steps
    U_new = ws.U[:,1:ws.U_ncols]*U_tilde[1:ws.U_ncols,1:r_new]
    V_new = ws.V[:,1:ws.V_ncols]*V_tilde[1:ws.V_ncols,1:r_new]

    # Create the truncated low-rank state
    state_new = State2D(U_new, S_new, V_new, r_new)

    return state_new
end

