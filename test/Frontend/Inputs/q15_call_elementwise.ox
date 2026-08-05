def biased(x: tensor[q15,16]) -> tensor[q15,16]:
  return shift(offset(x, bias=1024), amount=-2, rounding=nearest_ties_positive)

def q15_call_elementwise(x: tensor[q15,16]) -> tensor[q15,16]:
  return biased(x)
