using LinearAlgebra

"""
    diffcheb(n,xspan)

Compute Chebyshev differentiation matrices on `n`+1 points in the
interval `xspan`. Returns a vector of nodes and the matrices for the
first and second derivatives.
"""
function diffcheb(n,xspan)
    x = [ -cos( k*π/n ) for k in 0:n ]    # nodes in [-1,1]
    
    # Off-diagonal entries.
    c = [2; ones(n-1); 2];    # endpoint factors
    dij = (i,j) -> (-1)^(i+j)*c[i+1]/(c[j+1]*(x[i+1]-x[j+1]))
    Dₓ = [ dij(i,j) for i in 0:n, j in 0:n ]

    # Diagonal entries.
    Dₓ[isinf.(Dₓ)] .= 0         # fix divisions by zero on diagonal
    s = sum(Dₓ,dims=2)
    Dₓ -= diagm(s[:,1])         # "negative sum trick"

    # Transplant to [a,b].
    a,b = xspan
    x = @. a + (b-a)*(x+1)/2
    Dₓ = 2*Dₓ/(b-a)             # chain rule

    # Second derivative.
    Dₓₓ = Dₓ^2
    return x,Dₓ,Dₓₓ
end