def basic_product(x: f32, y: f32) -> f32:
  return x * y

def basic_f32(x: f32, y: f32, z: f32, w: f32, d: f32) -> f32:
  positive = +y
  product = basic_product(x, positive)
  total = product + z
  return -(total - w) / d
