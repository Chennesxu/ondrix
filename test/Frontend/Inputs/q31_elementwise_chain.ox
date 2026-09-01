def q31_elementwise_chain(x: tensor[q31,32], y: tensor[q31,32]) -> tensor[q31,32]:
  return add(mult(x, y),
             shift(negate(abs(offset(sub(x, y), bias=1073741824))),
                   amount=-31, rounding=nearest_ties_positive),
             overflow=wrap)
