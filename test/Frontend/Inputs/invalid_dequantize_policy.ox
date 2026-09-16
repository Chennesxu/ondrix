def invalid_dequantize_policy(x: tensor[q31,64]) -> tensor[f32,64]:
  return dequantize(x, to=f32, rounding=nearest_even)
