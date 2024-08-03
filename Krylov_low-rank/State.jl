"""
Struct to hold the factors of a low-rank object in 2D, i.e., A = USV'.
"""
struct State2D{T1<: AbstractMatrix, T2 <:AbstractMatrix}
    U::T1
    S::T2
    V::T1
    r::Int
end
