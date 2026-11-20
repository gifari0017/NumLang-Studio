# NumLang Studio: a tour of the methods.
compare root x^3 - 2x - 5 in 2 3
compare solve [2 1 -1; -3 -1 2; -2 1 2] = [8 -11 -3]
compare ode y' = y, y(0) = 1, h = 0.1, to 1 exact exp(x)

scene curtain
  cloth 13 x 9 spacing 0.05
  stiffness 150
  wind 0
  integrator rk4
  h = 4 ms
end
