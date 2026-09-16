def q15_literals_calls(x: tensor[q15,32], y: tensor[q15,32]) -> tensor[q15,32]:
  return sub(offset(add(gain(y, gain=3), gain(x, gain=16384)), bias=-1024),
             gain(x, gain=16384, rounding=nearest_even))
