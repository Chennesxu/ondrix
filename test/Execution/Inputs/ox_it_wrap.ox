def it_wrap(x: tensor[i32, 8], y: tensor[i32, 8], k: i32) -> tensor[i32, 8]:
  return -(x * y + k) * 3 - x
