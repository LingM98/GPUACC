include("State.jl")
include("Workspace.jl")
include("SolverParameters.jl")

include("bases.jl")
include("sylvester.jl")

"""
Function that approximately solves the Sylvester equation
    A1 X + X A2' + B = 0
Using an extended Krylov subspace approach.

Input:
    state_old: SVD factors from the previous time level
    ws: A reusable workspace for the Krylov solver
    params: A parameter struct containing information about the solver (tol, rank, backend, etc.)

Output:
    state_new: SVD factors for the new time level

The iteration terminates early provided the following condition is satisfied:
    ||A1 X + X A2'- B|| < ||B|| * tol,
where the norms are applied in a spectral sense. The term on the RHS is computed 

Note: The residual is measured by projecting onto the low-dimensional subspaces to reduce the
complexity of its formation.
"""
function extended_krylov_step!(state_old::State2D, ws::ExtendedKrylovWorkspace2D, params::SolverParameters)

    # Retrieve some input parameters
    max_iter, rel_tol, backend = params.max_iter, params.rel_tol, params.backend

    # Compute the threshold used to construct the Krylov basis
    threshold = get_threshold(state_old.S, rel_tol, backend)

    # Precompute the LU factorizations of A1 and A2
    compute_LU_factorizations!(ws)

    # Storage for the Krylov bases
    initialize_bases!(ws, state_old)

    # Variables to track during construction of the Krylov subspaces
    converged = false
    num_iterations = 0

    for iter_count = 1:max_iter

        update_bases_and_orthogonalize!(ws, backend)

        residual_norm = build_and_solve_sylvester!(state_old, ws, backend)

        if residual_norm < threshold
            num_iterations = iter_count
            converged = true
            break
        end

        shuffle_iterates!(ws)

    end

    state_new = apply_svd_truncation!(ws, params)

    return state_new, num_iterations
end

