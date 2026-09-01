def q31_matmul(a: tensor[q31,4,64], b: tensor[q31,64,3]) -> tensor[q31,4,3]:
  return matmul(a, b)
