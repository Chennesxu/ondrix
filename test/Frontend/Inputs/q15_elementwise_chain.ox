def q15_elementwise_chain(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  return add(mult(x, y),
             shift(negate(abs(offset(sub(x, y), bias=1024))),
                   amount=-2, rounding=nearest_ties_positive),
             overflow=wrap)
