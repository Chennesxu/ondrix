def sl_win(x: buffer[f32], h: buffer[f32], y: mut buffer[f32]) -> f32:
  for n in range(len(y)):
    y[n] = fir(x[n:n + len(h)], h, contract=off)
  return rms(x[2:10], contract=off) + dot(y, y, contract=off)
