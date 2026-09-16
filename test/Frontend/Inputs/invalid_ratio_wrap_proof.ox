def invalid_ratio_wrap_proof(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  return x / offset(abs(y), bias=1, overflow=wrap)
