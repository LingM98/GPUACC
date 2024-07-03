"""
Approximately solve
    A X + X B' + Vx_n*S_n*Vy_n' = 0

Input:
    A: coeff matrix,
    B: coeff matrix,
    Vx_n, S_n, and Vy_n: SVD factors of initial matrix data.
    tol stopping tolerance, with stopping criterion
          ||A X  + X B'-Vx_n*S_n*Vy_n'||
          ----------------------------------------  < tol
                   ||Vx_n*S_n*Vy_n'||
    computed in a cheap manner.

Output:
    solution factor   X = Vx_nn*S_nn*Vy_nn'
"""
@fastmath function sylvester_extended_krylov(Vx_n, Vy_n, S_n, A, B, tol)

    # Precompute the norm of SVD from the previous level
    normb = norm(S_n)

    # Precompute the factorization of A and B
    FA = factorize(A)
    FB = factorize(B)

    DxVxn = Vx_n
    DyVyn = Vy_n

    Dxinv = Vx_n
    Dyinv = Vy_n

    # Create these variables for storing output
    S1 = S_n
    Vx_nn = Vx_n
    Vy_nn = Vy_n
    S_nn = S_n

    # Variables to track during construction of the Krylov subspaces
    converged = false
    iter_count = 0

    # Tolerance for the construction of the Krylov basis
    threshold = normb*tol

    while !converged

        # Construct the extended Krylov bases
        DxVxn = A*DxVxn
        DyVyn = B*DyVyn

        Dxinv = FA\Dxinv
        Dyinv = FB\Dyinv

        # Enrich the basis used for the Krylov subspaces
        if iter_count == 0
            
            Vx_aug = hcat(Vx_n, DxVxn, Dxinv)
            Vy_aug = hcat(Vy_n, DyVyn, Dyinv)

        else
            
            Vx_aug = hcat(Vx_nn, DxVxn, Dxinv)
            Vy_aug = hcat(Vy_nn, DyVyn, Dyinv)

        end

        # Orthogonalize the augmented bases
        Vx_nn, _ = qr(Vx_aug)
        Vy_nn, _ = qr(Vy_aug)

        # Build and solve the reduced system using the Sylvester solver
        AV = A*Vx_nn
        A11 = Vx_nn'*AV
        BV = B*Vy_nn
        B11 = Vy_nn'*BV    
        RHS = (Vx_nn'*Vx_n)*S_n*(Vy_n'*Vy_nn)

        S1 = sylvc(A11, B11, RHS)

        # Check convergence of the solver
        _, R11 = qr(hcat(Vx_nn, AV))
        _, R22 = qr(hcat(Vy_nn, BV))

        iter_count += 1

        residual = R11 * [-RHS S1; S1 zeros(size(S1, 1), size(S1, 2))] * R22'
        residual_norm = norm(residual)

        if residual_norm < threshold
            converged = true
            break
        end

    end

    t1, S1n, t2 = svd(S1)

    # Here S1n is a vector so we do this before
    # we promote S1n to a diagonal matrix
    r = findlast(x -> x > tol, S1n)
    S1n = Diagonal(S1n)

    # Use views to prevent unnecessary copies
    Vx_nn = Vx_nn*view(t1, :, 1:r)
    Vy_nn = Vy_nn*view(t2, :, 1:r)
    S_nn = view(S1n, 1:r, 1:r)

    return Vx_nn, Vy_nn, S_nn, iter_count

end


