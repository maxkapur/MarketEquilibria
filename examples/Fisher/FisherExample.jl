using MarketEquilibria
using DelimitedFiles
using Random
using Plots

form = [:linear, :CES, :CES, :CES, :cobb_douglas, :CES, :CES, :CES, :CES]
rho = [1, .75, .5, .25, 0, -1, -2, -5, -10]
rho_label = ["Linear", "0.75", "0.5", "0.25",
             "Cobb–\nDouglas", "-1.0", "-2.0", "-5.0", "-10.0"]
samp = size(rho)[1]

m, n = 4, 6

# endowments =  1 .+ randexp(m)
# A = rand(m, n)
# A ./= sum(A, dims=2)
# supplies = 1 .+ rand(n)
#
# writedlm("examples/Fisher/endowments.dat", endowments)
# writedlm("examples/Fisher/A.dat", A)
# writedlm("examples/Fisher/supplies.dat", supplies)

endowments = reshape(readdlm("examples/Fisher/endowments.dat"), :)
A          = readdlm("examples/Fisher/A.dat")
supplies   = reshape(readdlm("examples/Fisher/supplies.dat"), :)

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

plot(1:samp,
     prices',
     xticks=(1:samp, rho_label),
     xlabel="ρ",
     ylabel="price",
     title="Homogenous Fisher equilibria for CES utility")

savefig("examples/Fisher/EqPlot.png")
savefig("examples/Fisher/EqPlot.pdf")
