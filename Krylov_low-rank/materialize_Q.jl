"""
Wrapped calls to the thin QR decomposition. These calls convert the return Q from
a compact type to a dense matrix of the appropriate type.
"""

function materialize_Q(Q_factor::T, backend::CPU_backend) where {T}
   return Matrix(Q_factor) 
end

@static if @isdefined(CUDA)
    function materialize_Q(Q_factor::T, backend::CUDA_backend) where {T}
        return CuMatrix(Q_factor) 
    end
    materialize_Q(Q_factor::T, backend::CUDA_UVM_backend) where {T} = materialize_Q(Q_factor, CUDA_backend()) 
end
