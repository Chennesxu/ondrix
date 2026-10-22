def ix_traps(x: buffer[f32], y: mut buffer[f32], mode: i32):
  if mode == 0:
    n = len(x) * 1073741824 * 1073741824
    y[n / 1073741824 / 1073741824 - 1] = 1.0
  elif mode == 1:
    m = (0 - len(x)) * 1073741824 * 1073741824 * 8
    y[(0 - 1) * m] = 2.0
  elif mode == 2:
    m = (0 - len(x)) * 1073741824 * 1073741824 * 8
    y[m / (0 - 1)] = 3.0
  else:
    y[len(y) / len(x) - 1] = 4.0
