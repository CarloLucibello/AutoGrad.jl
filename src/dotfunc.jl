# These functions are used in derivatives for julia6 compatibility; @compat does not work with Rec args.

# getsyms(a)=[ (isa(x,Tuple) ? x[1] : x) for x in a ]

# for p in (number1arg, number1zero, math1arg, trig1arg)
#     for f in getsyms(p)

flist = [:abs,:abs2,:isinteger,:sign,:signbit,:cbrt,:deg2rad,:exp,:exp10,:exp2,:expm1,:log,:log10,:log1p,:log2,:rad2deg,:significand,:sqrt,:acos,:acosd,:acosh,:acot,:acotd,:acoth,:acsc,:acscd,:acsch,:asec,:asecd,:asech,:asin,:asind,:asinh,:atan,:atand,:atanh,:cos,:cosc,:cosd,:cosh,:cospi,:cot,:cotd,:coth,:csc,:cscd,:csch,:sec,:secd,:sech,:sin,:sinc,:sind,:sinh,:sinpi,:tan,:tand,:tanh]

for f in flist
    g = Symbol(f,"_dot")
    if VERSION >= v"0.6.0"
        @eval $g(x)=$f.(x)
    else
        @eval $g(x)=$f(x)
    end
end
