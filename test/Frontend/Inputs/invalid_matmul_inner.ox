def matmul_inner(a: tensor[q15,4,8], b: tensor[q15,7,3]) -> tensor[q15,4,3]:
  return matmul(a, b)
