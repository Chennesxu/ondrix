def invalid_matmul_raw_high_rounding(a: tensor[q31,4,16], b: tensor[q31,16,3]) -> tensor[q31,4,3]:
  return matmul(a, b, product=raw_high, product_rounding=nearest_even)
