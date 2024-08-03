#include("thin_qr.jl")

"""
Solves the continuous Sylvester equation on the CPU.
"""
@views function build_and_solve_sylvester!(state_old::State2D, ws::ExtendedKrylovWorkspace2D, backend::CPU_backend)

    # Unpack the previous low-rank state data
    U_old, S_old, V_old = state_old.U, state_old.S, state_old.V

    # Unpack some items from the workspace for convenience
    A1, A2 = ws.A1, ws.A2
    U, V = ws.U, ws.V
    U_ncols, V_ncols = ws.U_ncols, ws.V_ncols

    S1, A1_tilde, A2_tilde, B1_tilde = ws.S1, ws.A1_tilde, ws.A2_tilde, ws.B1_tilde
    A1U, A2V = ws.A1U, ws.A2V
    block_matrix, residual = ws.block_matrix, ws.residual

    # Temporaries for evaluating B1_tilde
    tmp1 = similar(B1_tilde, U_ncols, size(U_old,2))
    tmp2 = similar(B1_tilde, size(V_old,2), V_ncols)
    tmp3 = similar(B1_tilde, U_ncols, size(S_old,2))

    # Create a container to hold references to the two R matrices defined locally in the async blocks
    R_container = Vector{Matrix{eltype(U)}}(undef, 2)

    # Build and solve the reduced system using the Sylvester solver
    @sync begin
        A1U_eval = Threads.@spawn mul!(A1U[:,1:U_ncols], A1, U[:,1:U_ncols])

        A2V_eval = Threads.@spawn mul!(A2V[:,1:V_ncols], A2, V[:,1:V_ncols])
        
        A1_tilde_eval = Threads.@spawn begin
            wait(A1U_eval)
            mul!(A1_tilde[1:U_ncols,1:U_ncols], U[:,1:U_ncols]', A1U[:,1:U_ncols])
        end

        A2_tilde_eval = Threads.@spawn begin
            wait(A2V_eval)
            mul!(A2_tilde[1:V_ncols,1:V_ncols], V[:,1:V_ncols]', A2V[:,1:V_ncols])
        end

        tmp1_eval = Threads.@spawn mul!(tmp1, U[:,1:U_ncols]', U_old[:,:])
        tmp2_eval = Threads.@spawn mul!(tmp2, V_old[:,:]', V[:,1:V_ncols])
        
        tmp3_eval = Threads.@spawn begin
            wait(tmp1_eval)
            mul!(tmp3, tmp1, S_old)
        end

        B1_tilde_eval = Threads.@spawn begin
            for task in [tmp2_eval, tmp3_eval]
                wait(task)
            end
            mul!(B1_tilde[1:U_ncols,1:V_ncols], tmp3, tmp2)
        end

        Threads.@spawn begin
            wait(A1U_eval)
            _, RU = qr!(hcat(U[:,1:U_ncols], A1U[:,1:U_ncols]))
            R_container[1] = RU
        end

        Threads.@spawn begin
            wait(A2V_eval)
            _, RV = qr!(hcat(V[:,1:V_ncols], A2V[:,1:V_ncols]))
            R_container[2] = RV
        end

        sylvester_solve = Threads.@spawn begin
            # The Sylvester solve cannot be performed until all coefficients are available
            for task in [A1_tilde_eval, A2_tilde_eval, B1_tilde_eval]
                wait(task)
            end

            S1[1:U_ncols,1:V_ncols] .= sylvc(A1_tilde[1:U_ncols,1:U_ncols], A2_tilde[1:V_ncols,1:V_ncols], B1_tilde[1:U_ncols,1:V_ncols])
        end

        # Zero block in the residual
        Threads.@spawn block_matrix[U_ncols .+ (1:U_ncols),V_ncols .+ (1:V_ncols)] .= 0.0

        Threads.@spawn begin
            wait(B1_tilde_eval)
            block_matrix[1:U_ncols,1:V_ncols] .= -B1_tilde[1:U_ncols,1:V_ncols]
        end

        # Off diagonal blocks in the residual
        Threads.@spawn begin
            wait(sylvester_solve)
            block_matrix[U_ncols .+ (1:U_ncols),1:V_ncols] .= S1[1:U_ncols,1:V_ncols]
            block_matrix[1:U_ncols,V_ncols .+ (1:V_ncols)] .= S1[1:U_ncols,1:V_ncols]
        end
    end

    # Access the triangular matrices from the QR factorizations
    RU = R_container[1]
    RV = R_container[2]

    # Create a reference to the relevant block of the residual
    residual_block = residual[1:size(RU,1),1:size(RV,2)]
    residual_block .= RU[:,:]*block_matrix[1:(2*U_ncols),1:(2*V_ncols)]*RV[:,:]'

    # Evaluate the spectral norm of the residual block
    return opnorm(residual_block,2)
end


@static if @isdefined(CUDA)

    @views function build_and_solve_sylvester!(state_old::State2D, ws::ExtendedKrylovWorkspace2D, backend::CUDA_backend)

        # Unpack the previous low-rank state data
        U_old, S_old, V_old = state_old.U, state_old.S, state_old.V

        # Unpack some items from the workspace for convenience
        A1, A2 = ws.A1, ws.A2
        U, V = ws.U, ws.V
        U_ncols, V_ncols = ws.U_ncols, ws.V_ncols

        S1, A1_tilde, A2_tilde, B1_tilde = ws.S1, ws.A1_tilde, ws.A2_tilde, ws.B1_tilde
        A1U, A2V = ws.A1U, ws.A2V
        block_matrix, residual = ws.block_matrix, ws.residual

        # Create a container to hold references to the two R matrices defined locally in the async blocks
        R_container = Vector{CuMatrix{eltype(U)}}(undef, 2)

        # Compute the coefficients of the Sylvester equation in parallel and copy them using
        # different streams on the device
        @sync begin
            A1U_eval = Threads.@spawn mul!(A1U[:,1:U_ncols], A1, U[:,1:U_ncols])
            A2V_eval = Threads.@spawn mul!(A2V[:,1:V_ncols], A2, V[:,1:V_ncols])

            A1_tilde_DtH = Threads.@spawn begin
                wait(A1U_eval)
		        A1_tilde[1:U_ncols,1:U_ncols] .= Array(U[:,1:U_ncols]'*A1U[:,1:U_ncols])
	        end

            A2_tilde_DtH = Threads.@spawn begin
                wait(A2V_eval)
		        A2_tilde[1:V_ncols,1:V_ncols] .= Array(V[:,1:V_ncols]'*A2V[:,1:V_ncols])
            end

            B1_tilde_DtH = Threads.@spawn begin
                # Here we set the block = B1 (but multiply by -1 later) to reuse memory
                block_matrix[1:U_ncols,1:V_ncols] .= (U[:,1:U_ncols]'*U_old[:,:])*S_old[:,:]*(V_old[:,:]'*V[:,1:V_ncols])
		        B1_tilde[1:U_ncols,1:V_ncols] .= Array(block_matrix[1:U_ncols,1:V_ncols])
                block_matrix[1:U_ncols,1:V_ncols] .*= -1.0
            end

            Threads.@spawn begin
                wait(A1U_eval)
		        _, RU = qr!(hcat(U[:,1:U_ncols], A1U[:,1:U_ncols]))
                R_container[1] = RU
            end

            Threads.@spawn begin
                wait(A2V_eval)
                _, RV = qr!(hcat(V[:,1:V_ncols], A2V[:,1:V_ncols]))
                R_container[2] = RV
            end

            sylvester_solve = Threads.@spawn begin
                # The Sylvester solve cannot be performed until all copies are completed
                for task in [A1_tilde_DtH, A2_tilde_DtH, B1_tilde_DtH]
                    wait(task)
                end

                # Solve the Sylvester equation on the CPU and then copy the result back to the GPU
                S1[1:U_ncols,1:V_ncols] .= sylvc(A1_tilde[1:U_ncols,1:U_ncols], A2_tilde[1:V_ncols,1:V_ncols], B1_tilde[1:U_ncols,1:V_ncols])
            end

            Threads.@spawn block_matrix[U_ncols .+ (1:U_ncols),V_ncols .+ (1:V_ncols)] .= 0.0

            # The remaining tasks depend on the solution of the Sylvester equation
            Threads.@spawn begin
                wait(sylvester_solve)
		        S1_d = CuArray(S1[1:U_ncols,1:V_ncols])
		        block_matrix[U_ncols .+ (1:U_ncols),1:V_ncols] .= S1_d
		        block_matrix[1:U_ncols,V_ncols .+ (1:V_ncols)] .= S1_d
            end
        end

        # Access the triangular matrices from the QR factorizations
        RU = R_container[1]
        RV = R_container[2]

        # Check convergence of the solver using the spectral norm of the residual
        # RU*[-B1_tilde S1; S1 zeros(size(S1, 1), size(S1, 2))]*RV'
        residual_block = residual[1:size(RU,1),1:size(RV,2)]
        residual_block .= RU[:,:]*block_matrix[1:(2*U_ncols),1:(2*V_ncols)]*RV[:,:]'

        # Evaluate the spectral norm of the residual block
	tmp = copy(residual_block)
        sigma = svdvals!(tmp)
        @CUDA.allowscalar spectral_norm = sigma[1]

        return spectral_norm
    end

    @views function build_and_solve_sylvester!(state_old::State2D, ws::ExtendedKrylovWorkspace2D, backend::CUDA_UVM_backend)

        # Unpack the previous low-rank state data
        U_old, S_old, V_old = state_old.U, state_old.S, state_old.V

        # Unpack some items from the workspace for convenience
        A1, A2 = ws.A1, ws.A2
        U, V = ws.U, ws.V
        U_ncols, V_ncols = ws.U_ncols, ws.V_ncols

        S1, A1_tilde, A2_tilde, B1_tilde = ws.S1, ws.A1_tilde, ws.A2_tilde, ws.B1_tilde
        A1U, A2V = ws.A1U, ws.A2V
        block_matrix, residual = ws.block_matrix, ws.residual

         # Create a container to hold references to the two R matrices defined locally in the async blocks
         R_container = Vector{CuMatrix{eltype(U)}}(undef, 2)

         # Compute the coefficients of the Sylvester equation in parallel and copy them using
         # different streams on the device
         @sync begin
            A1U_eval = Threads.@spawn mul!(A1U[:,1:U_ncols], A1, U[:,1:U_ncols])
            A2V_eval = Threads.@spawn mul!(A2V[:,1:V_ncols], A2, V[:,1:V_ncols])
 
             A1_tilde_DtH = Threads.@spawn begin
                A1_tilde = unsafe_wrap(CuArray, A1_tilde)
                mul!(A1_tilde[1:U_ncols,1:U_ncols], U[:,1:U_ncols]', A1U[:,1:U_ncols])
                A1_tilde = unsafe_wrap(Array, A1_tilde)              
             end
 
             A2_tilde_DtH = Threads.@spawn begin
                A2_tilde = unsafe_wrap(CuArray, A2_tilde)
                mul!(A2_tilde[1:V_ncols,1:V_ncols], V[:,1:V_ncols]', A2V[:,1:V_ncols])
                A2_tilde = unsafe_wrap(Array, A2_tilde)  
             end
 
             B1_tilde_DtH = Threads.@spawn begin
                B1_tilde = unsafe_wrap(CuArray, B1_tilde)
                B1_tilde[1:U_ncols,1:V_ncols] .= (U[:,1:U_ncols]'*U_old[:,:])*S_old[:,:]*(V_old[:,:]'*V[:,1:V_ncols])
                block_matrix[1:U_ncols,1:V_ncols] .= -B1_tilde[1:U_ncols,1:V_ncols]
                B1_tilde = unsafe_wrap(Array, B1_tilde) 
             end
 
             Threads.@spawn begin
                 wait(A1U_eval)
                 _, RU = qr!(hcat(U[:,1:U_ncols], A1U[:,1:U_ncols]))
                 R_container[1] = RU
             end
 
             Threads.@spawn begin
                 wait(A2V_eval)
                 _, RV = qr!(hcat(V[:,1:V_ncols], A2V[:,1:V_ncols]))
                 R_container[2] = RV
             end
 
             sylvester_solve = Threads.@spawn begin
                 # The Sylvester solve cannot be performed until all copies are completed
                 for task in [A1_tilde_DtH, A2_tilde_DtH, B1_tilde_DtH]
                     wait(task)
                 end

                 S1 = unsafe_wrap(Array, S1)
                 
                 # Solve the Sylvester equation on the CPU and then copy the result back to the GPU
                 S1[1:U_ncols,1:V_ncols] .= sylvc(A1_tilde[1:U_ncols,1:U_ncols], A2_tilde[1:V_ncols,1:V_ncols], B1_tilde[1:U_ncols,1:V_ncols])  
             end
 
             Threads.@spawn block_matrix[U_ncols .+ (1:U_ncols),V_ncols .+ (1:V_ncols)] .= 0.0
 
             # The remaining tasks depend on the solution of the Sylvester equation
             Threads.@spawn begin
                 wait(sylvester_solve)
                 S1 = unsafe_wrap(CuArray, S1)
                 copyto!(block_matrix[U_ncols .+ (1:U_ncols),1:V_ncols], S1[1:U_ncols,1:V_ncols])
                 copyto!(block_matrix[1:U_ncols,V_ncols .+ (1:V_ncols)], S1[1:U_ncols,1:V_ncols]) 
             end
         end
 
         # Access the triangular matrices from the QR factorizations
         RU = R_container[1]
         RV = R_container[2]
 
         # Check convergence of the solver using the spectral norm of the residual
         # RU*[-B1_tilde S1; S1 zeros(size(S1, 1), size(S1, 2))]*RV'
         residual_block = residual[1:size(RU,1),1:size(RV,2)]
         residual_block .= RU[:,:]*block_matrix[1:(2*U_ncols),1:(2*V_ncols)]*RV[:,:]'
 
        # Evaluate the spectral norm of the residual block
	tmp = copy(residual_block)
        sigma = svdvals!(tmp)
        @CUDA.allowscalar spectral_norm = sigma[1]

        return spectral_norm
    end

end

