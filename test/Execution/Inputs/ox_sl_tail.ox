def sl_tail(x: tensor[f32, 8], a: i64) -> f32:
  w = x[a:]
  s = 0.0
  for i in range(len(w)):
    s = s + w[i]
  return s
