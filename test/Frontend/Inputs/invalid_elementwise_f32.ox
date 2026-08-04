def invalid_elementwise_f32(x: tensor[f32,32], y: tensor[f32,32]) -> tensor[f32,32]:
  return add(x, y)
