module MarketEquilibria

using Random
using JuMP
using Ipopt

export homogeneousfisher

"""
    homogeneousfisher(endowments, A, supplies)

§6.2: Fisher Model with Homogeneous Consumers. `form` specifies which utility
function consumers use.
"""
function homogeneousfisher(endowments::Array{Float64,1}, A::Array{Float64,2}, supplies::Array{Float64,1}, form=:linear, rho=0.5)
    @assert form in [:linear, :CES, :cobb_douglas, :leontief] "Unknown form"
    if form == :cobb_douglas
        rho = 1e-8          # Ideally 0
    elseif form == :leonteif
        rho = -10           # Ideally -∞
    end

    (n, m) = size(A)
    @assert size(endowments) == (n, ) "Dim mismatch between A and endowments"
    @assert size(supplies) == (m, ) "Dim mismatch between A and supplies"

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)

    @variable(model, X[1:n, 1:m], lower_bound=0)
    @constraint(model, Supply[j in 1:m], sum(X[:, j]) ≤ supplies[j])

    if form==:linear
        # Very bad results if you allow the default scaling 🐸☕️
        set_optimizer_attribute(model, "nlp_scaling_method", "none")
        @NLobjective(model, Max, sum(endowments[i] * log(sum(A[i, j] * X[i, j] for j in 1:m)) for i in 1:n))
    else
        @NLobjective(model, Max, sum(endowments[i] * log(sum(A[i, j] * X[i, j] ^ rho for j in 1:m) ^ (1 / rho)) for i in 1:n))
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


end
