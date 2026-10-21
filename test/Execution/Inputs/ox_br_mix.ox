def br_mix(x: tensor[f32, 8], c: bool) -> tensor[f32, 8]:
  y = x
  if c:
    y = x * 2.0
  else:
    y = x - 1.0
  return y
