def invalid_f32_gain_denominator(input: tensor[f32,16]) -> tensor[f32,16]:
  return gain(input, gain=[1, 0], contract=off)
