def f32_dct(x: tensor[f32,8]) -> tensor[f32,8]:
  return dct(x, contract=fma)
