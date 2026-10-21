def br_clip(x: buffer[f32], y: mut buffer[f32], lo: f32, hi: f32) -> i32:
  n = 0
  for i in range(len(x)):
    v = x[i]
    if v < lo:
      v = lo
      n = n + 1
    elif v <= hi:
      n = n + 100
    else:
      v = hi
    y[i] = v
  return n
