def helper(x: tensor[q15,64]) -> tensor[q15,64]:
  return gain(x, gain=19661)

def helper(x: tensor[q15,64]) -> tensor[q15,64]:
  return gain(x, gain=19661)

def invalid_call_duplicate(x: tensor[q15,64]) -> tensor[q15,64]:
  return helper(x)
