def invalid_f32_gain_constant(input: tensor[f32,16]) -> tensor[f32,16]:
  return gain(input, gain=[1, 0], contract=off)
