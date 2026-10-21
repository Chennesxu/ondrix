def out_scale(x: buffer[f32], y: mut buffer[f32], g: f32):
  for i in range(len(y)):
    y[i] = g * x[i]
