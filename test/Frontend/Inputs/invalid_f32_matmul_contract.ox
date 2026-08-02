def invalid_f32_matmul_contract(a: tensor[f32,2,2], b: tensor[f32,2,2]) -> tensor[f32,2,2]:
  return matmul(a, b, contract=exact)
