def it_quant(x: tensor[f32, 16]) -> tensor[i32, 16]:
  return convert(x, to=i32, rounding=nearest_even)
