def q31_matmul_floor(a: tensor[q31,4,64], b: tensor[q31,64,3]) -> tensor[q31,4,3]:
  return matmul(a, b, product_rounding=toward_negative, rounding=nearest_ties_positive)
