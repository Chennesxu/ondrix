def q15_infix_grouping(x: tensor[q15,16], y: tensor[q15,16], z: tensor[q15,16]) -> tensor[q15,16]:
  t = (x + y) * z
  return t - x - mult(y, z, overflow=wrap)
