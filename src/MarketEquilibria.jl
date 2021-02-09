module MarketEquilibria

using Random
using JuMP
using Ipopt

export homogeneousfisher, exchange

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
                           ρ::Float64=0.5)::Tuple{Array{Float64,2},Array{Float64,1}}
    @assert form in [:linear, :CES, :cobb_douglas, :leontief] "Unknown form"
    if form == :cobb_douglas
        ρ = 1e-8          # Ideally 0
    elseif form == :leonteif
        ρ = -10           # Ideally -∞
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
    else
        @NLobjective(model, Max, sum(endowments[i] * log(sum(A[i, j] * X[i, j] ^ ρ for j in 1:m) ^ (1 / ρ)) for i in 1:n))
    end

        # Explicit Cobb–Douglas; very poorly conditioned. I couldn't get it to converge.
        # @NLobjective(model, Max, sum(endowments[i] * log(prod(X[i, j] ^ A[i, j] for j in 1:m)) for i in 1:n))

        # Explicit Leontief is as follows, but minimum() not allowed by JuMP.
        # @NLobjective(model, Max, sum(endowments[i] * log(minimum(A[i, j] * X[i, j] for j in 1:m)) for i in 1:n))

        # Attempt at "softmin" differential substitute; returns errors.
        # @NLobjective(model, Max, sum(endowments[i] * log(-log(sum(exp(-A[i, j] * X[i, j]) for j in 1:m))) for i in 1:n))

    optimize!(model)
    return value.(X), dual.(Supply)
end


"""
    exchange(endowments, A, ρ=:linear)

§6.4: Convex programs for linear and CES exchange economies. Note that unlike `homogeneousfisher()`,
here `ρ` is a vector of CES parameters used by each player. Or pass `ρ=:linear` for linear model.
"""
function exchange(endowments::Array{Float64,2},
                  A::Array{Float64,2},
                  ρ::Union{Array{Float64,1},Symbol}=:linear)::Tuple{Array{Float64,1},Array{Float64,2}}
    (n, m) = size(A)
    @assert size(endowments) == (n, m)         "Dim mismatch between A and endowments"

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)

    if ρ == :linear
        @variable(model, ψ[1:m])
        @variable(model, X[1:n, 1:m], lower_bound=0)
        @NLconstraint(model, IndRat[i in 1:n, j in 1:m], sum(A[i, k] * X[i, k] for k in 1:m) ≥
                                                         A[i, j] * sum(endowments[i, k] * exp(ψ[k] - ψ[j])
                                                                       for k in 1:m))
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


end
