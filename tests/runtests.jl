using Test
using Random
using MarketEquilibria

@testset "Homogeneous Fisher model" begin
    samp = 5

    @testset "Linear" begin
        for _ in 1:samp
            (n, m) = rand(10:50, 2)

            endowments =  1 .+ randexp(n)
            A = rand(n, m)
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
            @test -endowments ≈ X * prices atol=1e-5
        end
    end

    @testset "Leontief" begin
        for _ in 1:samp
            (n, m) = rand(10:50, 2)

            endowments =  1 .+ randexp(n)
            A = rand(n, m)
            supplies = 1 .+ rand(m)

            A ./= sum(A, dims=2)
            X, prices = homogeneousfisher(endowments, A, supplies, :leontief)
            @test -endowments ≈ X * prices atol=1e-5
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
            println("ρ = $ρ")
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


@testset "CES Exchange economies" begin
    samp = 10

    for _ in 1:samp
        (n, m) = rand(10:50, 2)

        endowments =  1 .+ randexp(n, m)
        ρ = -rand(n)
        A = rand(n, m)

        prices, demands = elasticexchange(endowments, ρ, A)

        @test all(demands * prices .≥ endowments * prices)
    end
end
