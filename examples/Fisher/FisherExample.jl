using MarketEquilibria
using DelimitedFiles
using Random
using Plots

form = [:linear, :CES, :CES, :CES, :cobb_douglas,
        :CES, :CES, :CES, :CES, :CES, :leontief]
# Rho value ignored for anything other than :CES
rho = [000, .75, .5, .25, 000, -0.5,
       -1., -2, -5, -10, 000]
rho_label = ["Linear", "0.75", "0.5", "0.25",
             "Cobb–\nDouglas", "-0.5", "-1.0", "-2.0",
             "-5.0", "-10.0", "Leontief"]
samp = size(rho)[1]

m, n = 4, 6

# endowments =  1 .+ randexp(m)
# A = rand(m, n)
# A ./= sum(A, dims=2)
# supplies = 1 .+ rand(n)
#
# writedlm("examples/Fisher/in/endowments.dat", endowments)
# writedlm("examples/Fisher/in/A.dat", A)
# writedlm("examples/Fisher/in/supplies.dat", supplies)

endowments = reshape(readdlm("examples/Fisher/in/endowments.dat"), :)
A          = readdlm("examples/Fisher/in/A.dat")
supplies   = reshape(readdlm("examples/Fisher/in/supplies.dat"), :)

prices = zeros(n, samp)
X      = zeros(m, n, samp)

for i in 1:samp
    global X
    global prices
    X[:, :, i], prices[:, i] = homogeneousfisher(endowments, A, supplies, form[i], rho[i])
    prices[:, i] .*= -1.
    signal = isapprox(endowments, X[:, :, i] * prices[:, i], atol=1e-3)
    @assert signal "Something went wrong; rho = $(rho[i])"
end

writedlm("examples/Fisher/out/X.dat", X)
writedlm("examples/Fisher/out/prices.dat", prices)

#A_bar = sum(A, dims=1) / m
A_bar = round.(A' * endowments, digits=4)

p = plot(1:samp,
         prices',
         xticks=(1:samp, rho_label),
         xlabel="ρ",
         ylabel="price",
         linestyle=[:dash :dot :solid],
         title="Equilibrium price for homogeneous CES utility, $m players, $n resources",
         titlefontsize=12,
         legend=:topleft,
         label=reshape(["π$i, ā = $(A_bar[i])" for i in 1:n], 1, :))

savefig(p, "examples/Fisher/EqPlot.png")
savefig(p, "examples/Fisher/EqPlot.pdf")

X1 = X[1, :, :] ./ sum(X[1, :, :], dims = 1)
X1A = round.(A[1, :], digits=4)

q = plot(1:samp,
         cumsum(X1, dims=1)'[:,end:-1:1],
         fill=0,
         # fillcolor=[:green :red],
         xticks=(1:samp, rho_label),
         xlabel="ρ",
         ylabel="Player 1’s optimal allocation",
         ls=:solid,
         lc=:black, lw=.5,
         title="Equilibrium allocation for homogeneous CES utility",
         titlefontsize=12,
         legend=:bottom,
         label=reshape(["x$i, a = $(X1A[i])" for i in 1:n], 1, :))

savefig(q, "examples/Fisher/XPlot.png")
savefig(q, "examples/Fisher/XPlot.pdf")
