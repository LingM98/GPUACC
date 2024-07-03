using LinearAlgebra

"""
    diffmat2(n,xspan)

Compute 2nd-order-accurate differentiation matrices on n+1 points
in the interval `xspan`. Returns a vector of nodes and the matrices
for the first and second derivatives.
"""
function diffmat2(n,xspan)
    a,b = xspan
    h = (b-a)/n
    x = [ a + i*h for i in 0:n ]   # nodes

    # Define most of Dₓ by its diagonals.
    dp = fill(0.5/h,n)        # superdiagonal
    dm = fill(-0.5/h,n)       # subdiagonal
    Dₓ = diagm(-1=>dm,1=>dp)

    # Fix first and last rows.
    Dₓ[1,1:3] = [-1.5,2,-0.5]/h
    Dₓ[n+1,n-1:n+1] = [0.5,-2,1.5]/h

    # Define most of Dₓₓ by its diagonals.
    d0 =  fill(-2/h^2,n+1)    # main diagonal
    dp =  ones(n)/h^2         # super- and subdiagonal
    Dₓₓ = diagm(-1=>dp,0=>d0,1=>dp)

    # Fix first and last rows.
    Dₓₓ[1,1:4] = [2,-5,4,-1]/h^2
    Dₓₓ[n+1,n-2:n+1] = [-1,4,-5,2]/h^2

    return x,Dₓ,Dₓₓ
end