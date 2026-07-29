def matmul_extent(a: tensor[q15,4,65], b: tensor[q15,65,3]) -> tensor[q15,4,3]:
  return matmul(a, b)
