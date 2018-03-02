include("header.jl")

@testset "primitives" begin
    @test gradcheck(x->chol(x'x), rand(3,3))
    
    for t in AutoGrad.alltests()
        #@show t
        @test gradcheck(eval(AutoGrad,t[1]), t[2:end]...)
    end

    for A  in (rand(2,2), rand(2,3), rand(3,2))
        @test gradcheck(x->svd(x)[1], A)
        @test gradcheck(x->svd(x)[2], A)
        @test gradcheck(x->svd(x)[3], A)
    end
    @test gradcheck(x->chol(x'x), rand(3,3))

    @test gradcheck(x->qr(x)[1], rand(3,3))
    @test gradcheck(x->qr(x)[2], rand(3,3))
    @test gradcheck(x->(y=qr(x); sum(y[1]+y[2])), rand(3,3))

    @test gradcheck(x->lq(x)[1], rand(3,3))
    @test gradcheck(x->lq(x)[2], rand(3,3))
    @test gradcheck(x->(y=lq(x); sum(y[1]+y[2])), rand(3,3))
end

nothing
