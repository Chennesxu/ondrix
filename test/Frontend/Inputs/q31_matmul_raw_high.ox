def q31_matmul_raw_high(a: tensor[q31,4,16], b: tensor[q31,16,3]) -> tensor[q31,4,3]:
  return matmul(a, b, product=raw_high)
