def f32_quantize_q31(x: tensor[f32,64]) -> tensor[q31,64]:
  return quantize(x, to=q31, rounding=toward_negative)
