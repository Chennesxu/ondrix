def invalid_matmul_raw_high_q15(a: tensor[q15,4,16], b: tensor[q15,16,3]) -> tensor[q15,4,3]:
  return matmul(a, b, product=raw_high)
