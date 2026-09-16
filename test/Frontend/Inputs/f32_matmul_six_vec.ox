def f32_matmul_six_vec(a: tensor[f32,2,4], b: tensor[f32,4,6]) -> tensor[f32,2,6]:
  return matmul(a, b, contract=off)
