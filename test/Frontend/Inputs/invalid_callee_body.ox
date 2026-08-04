def helper(x: tensor[q15,7]) -> tensor[q15,7]:
  return dct(x)

def invalid_callee_body(x: tensor[q15,7]) -> tensor[q15,7]:
  return helper(x)
