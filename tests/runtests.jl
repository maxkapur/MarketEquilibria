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
            X, prices = homogeneousfisher(endowments, A, supplies, :CES, rand())
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
