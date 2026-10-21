def out_order(y: mut buffer[q15], i: i64, a: q15, b: q15):
  y[i] = ratio(a, b, nonpositive=trap)
