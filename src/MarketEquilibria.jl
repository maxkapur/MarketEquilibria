module MarketEquilibria

using Random
using JuMP
using Ipopt

export homogeneousfisher, tatonnement, exchange, production

"""
    homogeneousfisher(endowments, A, supplies, form, ρ)

§6.2: Fisher Model with Homogeneous Consumers. `form` specifies which utility
function consumers use, among `:linear`, `:CES`, `:cobb_douglas`, and `:leontief`. `ρ` is the parameter for
`:CES`.
"""
function homogeneousfisher(endowments::Array{Float64,1},
                           A::Array{Float64,2},
                           supplies::Array{Float64,1},
                           form=:linear,
                           ρ::Union{Float64,Nothing}=0.5)::Tuple{Array{Float64,2},
                                                                 Array{Float64,1}}
    @assert form in [:linear, :CES, :cobb_douglas, :leontief] "Unknown form"
    if form == :cobb_douglas
        ρ = 1e-8          # Ideally 0
    end

    (n, m) = size(A)
    @assert size(endowments) == (n, ) "Dim mismatch between A and endowments"
    @assert size(supplies) == (m, ) "Dim mismatch between A and supplies"

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)

    @variable(model, X[1:n, 1:m], lower_bound=0)
    @constraint(model, Supply[j in 1:m], sum(X[:, j]) ≤ supplies[j])

    if form == :linear || ρ > .97
        if form != :linear
            @warn "ρ > .97 tends to fail, so assuming linear"
        end
        # Very bad results if you allow the default scaling 🐸☕️
        set_optimizer_attribute(model, "nlp_scaling_method", "none")
        @NLobjective(model, Max, sum(endowments[i] * log(sum(A[i, j] * X[i, j] for j in 1:m)) for i in 1:n))
    elseif form == :leontief
        @variable(model, t)
        @objective(model, Max, t)
        @NLconstraint(model, tLessThan[j in 1:m],
                    t ≤ sum(endowments[i] * log(A[i, j] * X[i, j]) for i in 1:n))
    else
        @NLobjective(model, Max, sum(endowments[i] * log(sum(A[i, j] * X[i, j] ^ ρ for j in 1:m) ^ (1 / ρ)) for i in 1:n))
    end

        # Explicit Cobb–Douglas; very poorly conditioned. I couldn't get it to converge.
        # @NLobjective(model, Max, sum(endowments[i] * log(prod(X[i, j] ^ A[i, j] for j in 1:m)) for i in 1:n))

        # Explicit Leontief is as follows, but minimum() not allowed by JuMP.
        # Got around this using the linearization trick.
        # @NLobjective(model, Max, sum(endowments[i] * log(minimum(A[i, j] * X[i, j] for j in 1:m)) for i in 1:n))

        # Attempt at "softmin" differential substitute; returns errors.
        # @NLobjective(model, Max, sum(endowments[i] * log(-log(sum(exp(-A[i, j] * X[i, j]) for j in 1:m))) for i in 1:n))

    optimize!(model)
    return value.(X), dual.(Supply)
end


"""
    tatonnement(endowments, A, ρ, π0;
                η=1e-4, δ=0.5, α=0.5, verbose=false, maxit=1000, tol=1e-5)

§6.3.1.B: The Discrete Tâtonnement Process. Simulate a bidding process in search
of equilibrium prices for CES exchange.

Price update step is quite different from the book: I *multiply* prices
entrywise by the excess demand: `π .*= exp.(δ * Z * k ^ -α)`. This works pretty well.

`η` is a lower bound on the prices before normalization.
"""
function tatonnement(endowments ::AbstractArray{<:AbstractFloat,2},
                     A          ::AbstractArray{<:AbstractFloat,2},
                     ρ          ::AbstractArray{<:AbstractFloat,1},
                     π0         ::Union{AbstractArray{<:AbstractFloat,1}, Nothing};
                     η          ::AbstractFloat=1e-4,
                     δ          ::AbstractFloat=0.5,
                     α          ::AbstractFloat=0.4,
                     verbose    ::Bool=false,
                     maxit      ::Int=2000,
                     tol        ::AbstractFloat=1e-5)::Tuple{Array{Float64,1},
                                                             Array{Float64,2}}

    (n, m) = size(A)
    @assert (n, m) == size(endowments)
    W = endowments      # sorry

    @assert (n, )  == size(ρ)
    @assert all(ρ .< 1)
    @assert all(0 .< (tol, η, δ, α) .< 1)

    if π0 == nothing
        π0 = rand(m)
    end

    @assert (m, )  == size(π0)

    total_supply = sum(W, dims=1)
    # Express utility as fractions of endowment
    A .*= total_supply
    # So we can normalize W
    W ./= total_supply

    σ = 1 ./ (1 .- ρ)
    π_LB = η / (4 * n)
    π_UB = 1 + π_LB
    π_LB_tight = η / (2 * n)

    π = copy(π0)
    D = zeros(Float64, n, m)
    Z = zeros(Float64, m)

    for k in 1:maxit
        # Oracle that computes player demand under CES (eq. 6.7, p. 149)
        for i in 1:n, j in 1:m
            D[i, j] = A[i, j] ^ σ[i] * sum(π[k] * W[i, k] for k in 1:m) /
                       (π[j] ^ σ[i] * sum((A[i, k] * π[k] ^ -ρ[i]) ^ σ[i] for k in 1:m))
        end

        for j in 1:m
            # Total supply, after normalization, is ones
            Z[j] = sum(D[i, j] for i in 1:n) - 1
        end

        if NaN in Z || all(-tol .< Z .< tol)
            break
        end

        if verbose
            println("Iteration $k")
            println("  Current prices: ", round.(π, digits=5))
            println("  Excess demand:  ", round.(Z, digits=5))
        end

        # This equilibrium step process is completely different from the book.
        # Just something that makes intuitive sense. Book is unclear on how to
        # calculate step size.
        if all(π_LB .≤ π .≤ π_UB)
            verbose ? println("  Step toward equilibrium") : nothing
            π .*= exp.(δ * Z * k ^ -α)
        else                             # Bottom of p. 144
            verbose ? println("  Regularization step") : nothing
            for j in 1:m
                if π[j] > 1
                    π[j] = 1
                elseif π[j] < π_LB_tight
                    π[j] = π_LB_tight
                end
            end
        end
    end

    return π ./ sum(π), D
end


"""
    exchange(endowments, A, ρ=:linear)

§6.4: Convex programs for linear and CES exchange economies. Note that unlike `homogeneousfisher()`,
here `ρ` is a vector of CES parameters used by each player. Or pass `ρ=:linear` for linear model.
"""
function exchange(endowments::Array{Float64,2},
                  A::Array{Float64,2},
                  ρ::Union{Array{Float64,1},Symbol}=:linear)::Tuple{Array{Float64,1},
                                                                    Array{Float64,2}}
    (n, m) = size(A)
    @assert size(endowments) == (n, m)         "Dim mismatch between A and endowments"

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)

    if ρ == :linear
        # set_optimizer_attribute(model, "nlp_scaling_method", "none")

        @variable(model, ψ[1:m])
        @variable(model, X[1:n, 1:m], lower_bound=0)
        @NLconstraint(model, IndividualRationality[i in 1:n, j in 1:m],
                        sum(A[i, k] * X[i, k] for k in 1:m) ≥
                        A[i, j] * sum(endowments[i, k] * exp(ψ[k] - ψ[j]) for k in 1:m))
        @constraint(model, Supply[j in 1:m], sum(X[i, j] for i in 1:n) ==
                                             sum(endowments[i, j] for i in 1:n))

        optimize!(model)

        prices = exp.(value.(ψ))
        demands = value.(X)

    else
        @assert size(ρ) == (n, )                   "Dim mismatch between A and ρ"
        @assert all(-1 .≤ ρ .< 0)                  "Need ρ ∈ [-1, 0)"

        @variable(model, σ[1:m], lower_bound=0)
        @NLconstraint(model, Supply[j in 1:m], sum(A[i, j] ^ (1 / (1 - ρ[i])) *
                                               sum(σ[k] ^ 2 * endowments[i, k] for k in 1:m) /
                                               (
                                                   σ[j] ^ ((ρ[i] - 2) / (1 - ρ[i])) *
                                                   sum(A[i, k] ^ (1 / (1 - ρ[i])) * σ[k] ^ ((-2 * ρ[i]) / (1 - ρ[i]))
                                                       for k in 1:m)
                                               )
                                               for i in 1:n) ≤ σ[j] * sum(endowments[i, j] for i in 1:n))

        optimize!(model)

        prices = value.(σ) .^ 2

        demands = zeros(n, m)
        for i in 1:n, j in 1:m
        demands[i, j] = A[i, j] ^ (1 / (1 - ρ[i])) * sum(prices[k] * endowments[i, k] for k in 1:m) /
                        (
                            prices[j] ^ (1 / 1 - ρ[i]) *
                            sum(A[i, k] ^ (1 / (1 - ρ[i])) * prices[k] ^ (-ρ[i] / (1 - ρ[i])) for k in 1:m)
                        )
        end
    end

    return prices, demands
end

"""
    exchange(endowments, A_consumers, A_producers,
             ρ_consumers, ρ_producers, o_producers)

§6.6: Models with production. `ρ_consumers` describes
each consumer's CES utility function, while `ρ_producers` describes the CES
production function. Convergence is very iffy, but best when `A_consumers` is
tall and entries of `ρ` vectors are not too close to 0 or 1.
"""
function production(endowments::Array{Float64,2},
                    A_consumers::Array{Float64,2},
                    A_producers::Array{Float64,2},
                    ρ_consumers::Array{Float64,1},
                    ρ_producers::Array{Float64,1},
                    o_producers::Array{Int,1})::Tuple{Array{Float64,1},
                                                      Array{Float64,1},
                                                      Array{Float64,2}}

    (n, m) = size(A_consumers)
    (l, ) = size(ρ_producers)

    @assert size(A_producers) == (l, m)   "Dim mismatch between A_consumers and A_producers"
    @assert size(endowments) == (n, m)    "Dim mismatch between A_consumers and endowments"
    @assert size(ρ_consumers) == (n, )    "Dim mismatch between A_consumers and ρ_consumers"
    @assert size(o_producers) == (l, )    "Dim mismatch between o and ρ_producers"
    @assert all(0 .< ρ_consumers .< 1)    "Need ρ ∈ (0, 1)"
    @assert all(0 .< ρ_producers .< 1)    "Need ρ ∈ (0, 1)"

    A = vcat(A_producers, A_consumers)
    w = hcat(endowments, zeros(n, l))
    ρ = vcat(ρ_consumers, ρ_producers)
    σ = 1 ./ (1 .- ρ)
    o = vcat(o_producers, m .+ (1:n))

    # Comparison idx since comparison forbidden in lincon
    O = [o[k] == j for k in 1:l, j in 1:m]

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "nlp_scaling_method", "none")

    @variable(model, ψ[1:m + n], lower_bound=0)
    @variable(model, x[1:m + n], lower_bound=0)
    @variable(model, Z[1:l + n, 1:m], lower_bound=0)
    @variable(model, q[1:l + n], lower_bound=0)

    @NLconstraint(model, ConsumerRationality[i in 1:n],                   # 6.10
                    exp(ψ[m + i]) * x[m + i] ≥ # or ==
                    sum(exp(ψ[j]) * w[i, j] for j in 1:m))
    @NLconstraint(model, ProducerSolvency[k in 1:l + n],                  # 6.11
                    q[k] ≤
                    sum(A[k, j] * x[j] ^ ρ[k] for j in 1:m) ^ (1 / ρ[k]))
    @NLconstraint(model, ProducerRationality[k in 1:l + n],               # 6.12
                    exp(ψ[o[k]] * (1 - σ[k])) ≥
                    sum(A[k, j] ^ σ[k] * exp(ψ[j] * (1 - σ[k])) for j in 1:m))
    @constraint(model, MarketClearing[j in 1:m],                          # 6.13
                    sum(Z[k, j] for k in 1:l + n) ≤
                    sum(w[i, j] for i in 1:n) + sum(O[k, j] * q[k] for k in 1:l))
                                            # Equiv
                                            # + sum((o[k] == j) * q[k] for k in 1:l + n))
    @constraint(model, Supply[i in 1:n], x[m + i] ≤ q[l + i])             # 6.14

    #=  Final constraint given in final paragraph of 154. This should be redundant, but
        aids computation.                   =#
    @NLconstraint(model, ZeroProfit[k in 1:l + n],
                    sum(exp(ψ[j]) * Z[k, j] for j in 1:m) == exp(ψ[o[k]]) * q[k])

    optimize!(model)
    prices = exp.(value.(ψ)[1:m])

    # Player utility, specifically, demand of each player for her own utility item.
    demands = value.(x)[m + 1:end]
    inputs = value.(Z)[l + 1:end, :]

    return prices, demands, inputs
end


end
