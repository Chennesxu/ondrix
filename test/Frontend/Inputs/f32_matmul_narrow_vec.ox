def f32_matmul_narrow_vec(a: tensor[f32,4,16], b: tensor[f32,16,3]) -> tensor[f32,4,3]:
  return matmul(a, b, contract=fma)
