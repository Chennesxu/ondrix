def viterbi_short(symbols: tensor[q15, 20]) -> tensor[u8, 1]:
  return viterbi_decode(symbols, constraint_length=3, polynomials=[7, 5])
