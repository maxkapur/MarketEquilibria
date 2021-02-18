using Test
using Random
using MarketEquilibria

@testset "Homogeneous Fisher model" begin
    samp = 5

    @testset "Linear" begin
        for _ in 1:samp
            (n, m) = rand(10:50, 2)

            endowments =  1 .+ randexp(n)
            A = .25 .+ 0.5 .* rand(n, m)    # Yields better conditioning
            supplies = 1 .+ rand(m)

            X, prices = homogeneousfisher(endowments, A, supplies, :linear)
            @test -endowments ≈ X * prices atol=1e-5
        end
    end

    @testset "Cobb–Douglas" begin
        for _ in 1:samp
            (n, m) = rand(10:50, 2)

            endowments =  1 .+ randexp(n)
            A = rand(n, m)
            supplies = 1 .+ rand(m)

            A ./= sum(A, dims=2)         # Required by Cobb–Douglas family
            X, prices = homogeneousfisher(endowments, A, supplies, :cobb_douglas)
            @test -endowments ≈ X * prices atol=1e-4
        end
    end

    @testset "Leontief" begin
        for _ in 1:samp
            (n, m) = rand(10:50, 2)

            endowments =  1 .+ randexp(n)
            A = .25 .+ 0.5 .* rand(n, m)
            supplies = 1 .+ rand(m)

            A ./= sum(A, dims=2)
            X, prices = homogeneousfisher(endowments, A, supplies, :leontief)
            @test -endowments ≈ X * prices atol=1e-4
        end
    end

    @testset "CES partial substitutes: ρ ∈ (0, 1)" begin
        for _ in 1:samp
            (n, m) = rand(10:50, 2)

            endowments =  1 .+ randexp(n)
            A = rand(n, m)
            supplies = 1 .+ rand(m)

            A ./= sum(A, dims=2)
            ρ = .97 * rand()        # Fails consistently for ρ  > .99, so the model uses
                                    # linear whenever ρ > .97.
            X, prices = homogeneousfisher(endowments, A, supplies, :CES, ρ)
            @test -endowments ≈ X * prices atol=1e-5
        end
    end

    @testset "CES complementarity fx: ρ < 0" begin
        for _ in 1:samp
            (n, m) = rand(10:50, 2)

            endowments =  1 .+ randexp(n)
            A = rand(n, m)
            supplies = 1 .+ rand(m)

            A ./= sum(A, dims=2)
            X, prices = homogeneousfisher(endowments, A, supplies, :CES, -randexp())
            @test -endowments ≈ X * prices atol=1e-5
        end
    end
end


@testset "Tâtonnement process for exchange economies" begin
    samp = 10

    for _ in 1:samp
        (n, m) = rand(5:10, 2)

        W = rand(n, m)      # Endowments
        A = rand(n, m)      # Utility params
        ρ = 1 .- randexp(n)

        prices, demands = tatonnement(W, A, ρ, nothing)

        @test W * prices ≈ demands * prices atol=1e-4
    end
end


@testset "Convex programs for exchange economies" begin
    samp = 10

    @testset "Linear" begin
        for _ in 1:samp
            (n, m) = rand(20:50, 2)

            endowments =  1 .+ randexp(n, m)
            A = randexp(n, m)
            A .*= (rand(n, m) .< .75)
            A ./= sum(A, dims=2)

            prices, demands = exchange(endowments, A, :linear)

            @test all(demands * prices .≥ endowments * prices .- 10e-2)
        end
    end

    @testset "CES" begin
        for _ in 1:samp
            (n, m) = rand(10:50, 2)

            endowments =  1 .+ randexp(n, m)
            A = rand(n, m)
            ρ = -rand(n)

            prices, demands = exchange(endowments, A, ρ)

            @test all(demands * prices .≥ endowments * prices .- 10e-4)
        end
    end
end


@testset "CES production and exchange" begin
    samp = 10

    for i in 1:samp
        n = rand(5:20)
        m = rand(4:n)       # m > n case doesn't converge well
        l = rand(3:m)       # l > m makes no sense

        endowments =  1 .+ randexp(n, m)
        A_consumers = rand(n, m)
        A_consumers ./= sum(A_consumers, dims=2)

        A_producers = rand(l, m)
        A_producers ./= sum(A_producers, dims=2)

        ρ_consumers = .1 .+ .8 * rand(n)
        ρ_producers = .1 .+ .8 * rand(l)

        # Index of what producers produce. Use perm here so none are redundant.
        o_producers = randperm(m)[1:l]

        # Can't use your output as an input. Unclear if this affects convergence.
        for (k, j) in enumerate(o_producers)
            A_producers
            A_producers[k, j] = 0
        end

        prices, demands, inputs = production(endowments, A_consumers, A_producers,
                                             ρ_consumers, ρ_producers, o_producers)

        @test all(inputs * prices .≈ endowments * prices)
    end
end
