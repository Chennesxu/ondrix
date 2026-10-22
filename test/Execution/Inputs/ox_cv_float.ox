def cv_float(y: mut buffer[f32], k: i32, w: i64):
  y[0] = convert(k, to=f32)
  y[1] = convert(w, to=f32)
