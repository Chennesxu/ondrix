def f32_correlation(input: tensor[f32,6], kernel: tensor[f32,3]) -> tensor[f32,4]:
  return correlation(input, kernel, contract=fma)
