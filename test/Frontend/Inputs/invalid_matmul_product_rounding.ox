def q15_matmul_product(a: tensor[q15,4,8], b: tensor[q15,8,3]) -> tensor[q15,4,3]:
  return matmul(a, b, product_rounding=nearest_even)
