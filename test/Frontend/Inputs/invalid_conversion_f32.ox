def invalid_conversion_f32(x: tensor[f32,32]) -> tensor[f32,32]:
  return narrow(x, to=q15)
