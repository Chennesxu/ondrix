def invalid_cosine_shape(phase: tensor[q15,16]) -> tensor[q15,8]:
  return cosine(phase)
