def br_window(x: mut buffer[f32, 4], c: bool):
  k = 0
  if c:
    k = 5
  x[0] = dot(x[k:4], x[k:4], contract=off)
