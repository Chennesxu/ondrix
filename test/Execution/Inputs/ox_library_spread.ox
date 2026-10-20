def sample(x: buffer[f32,8], w: buffer[f32,8]) -> f32:
  return fir(x, w, contract=off)

def spread(x: buffer[f32,8], w: buffer[f32,8]) -> f32:
  return sample(x, w) - rms(x, contract=off) * dot(w, w, contract=off)
