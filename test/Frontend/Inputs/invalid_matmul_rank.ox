def matmul_rank(a: tensor[q15,4], b: tensor[q15,4,3]) -> tensor[q15,4,3]:
  return matmul(a, b)
