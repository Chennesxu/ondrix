def it_gather(x: buffer[f32], p: buffer[i32], y: mut buffer[f32]):
  for i in range(len(y)):
    y[i] = x[p[i]]
