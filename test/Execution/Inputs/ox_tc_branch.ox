def tc_branch(x: tensor[f32, 16], n: i32) -> tensor[f32, 16]:
  y = x
  for i in range(n):
    if y[0] > 10.0:
      y = y * 0.5
    elif y[1] < 0.0:
      y = -y
    else:
      y = y + 1.0
    y = y - 0.25
  return y
