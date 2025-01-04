import math
def test():
    p = 0.9
    n = 25
    k = 13
    result = sum([math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1)])
    print(f"the probability of getting at least  {k} successes from a sample of size {n} is {result}\n")

test()