def f32_matmul(a: tensor[f32,3,4], b: tensor[f32,4,2]) -> tensor[f32,3,2]:
  return matmul(a, b, contract=fma)
