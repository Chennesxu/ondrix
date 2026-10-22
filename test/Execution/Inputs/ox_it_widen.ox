def it_widen(x: tensor[i32, 8], y: tensor[i64, 8]) -> tensor[f32, 8]:
  return convert(x, to=f32) * 0.5 + convert(y, to=f32)
