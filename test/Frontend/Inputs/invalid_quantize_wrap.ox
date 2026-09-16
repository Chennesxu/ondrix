def invalid_quantize_wrap(x: tensor[f32,64]) -> tensor[q15,64]:
  return quantize(x, to=q15, overflow=wrap)
