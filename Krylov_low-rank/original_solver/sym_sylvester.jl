"""
A solver for the Sylvester equation
    AX + XB = C,
where the coefficient matrices A and B are assumed
to be symmetric and may or may not be sparse.

The new problem to solve is
    Lambda_A*X_tilde + X_tilde*Lambda_B, 
where 
    X_tilde = V_A'*X*V_B,
with Lambda_A and Lambda_B being diagonal matrices.
"""
@fastmath @views function sylvc(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
#
# To-do: Specialize this function to work with CUDA arrays? Also, check the CUDA.jl support
# for eigen. In particular, is this operation supported when A and B are sparse? Are the inputs always dense, or
# do we even need to worry about this?
#
    m, n = size(C);
    [m; n] == LinearAlgebra.checksquare(A, B) || throw(DimensionMismatch("A, B and C have incompatible dimensions"))

    # First compute the eigenvalue decompositions of A and B
    # They are guaranteed to be diagonalizable, since they are symmetric.
    A_eig = eigen(A)
    B_eig = eigen(B)

    lambda_A = A_eig.values
    V_A = A_eig.vectors

    lambda_B = B_eig.values
    V_B = B_eig.vectors  

    # Transform the problem into one which is diagonal, starting with the RHS
    B_tilde = V_A'*B*V_B

    # Solve for X_tilde using broadcasting operations
    X_tilde = B_tilde./(lambda_A .+ lambda_B')

    # Transform the solution to recover x
    X = V_A*X_tilde*V_B'

    return X

end



