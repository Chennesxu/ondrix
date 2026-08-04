def helper(x: tensor[q15,32], c: tensor[q15,8]) -> tensor[q15,25]:
  return fir_filter(x, c, boundary=valid)

def invalid_call_shape(x: tensor[q15,64], c: tensor[q15,8]) -> tensor[q15,57]:
  return helper(x, c)
