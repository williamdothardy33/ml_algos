import math
import numpy as np
import matplotlib.pyplot as plt
# Suppose we have a simple learning model whose growth function
# is m_h(N) = N + 1, hence d_vc = 1. Use the VC bound (2.12
# E_out(g) <= E_in(g) + math.sqrt((8 /  N) * math.log(4 * m_h(2 * N) / delta)))
# to estimate the probability that E_out will be within 0.1 of E_in given 100
# training examples [Hint: The estimate will be rediculous]

# 0.1 =  math.sqrt((8 /  N) * math.log(4 * m_h(2 * N) / delta)) 
# => math.pow(0.1, 2) = (8 /  N) * math.log(4 * m_h(2 * N) / delta)
# => (N / 8) * math.pow(0.1, 2) = math.log(4 * m_h(2 * N) / delta)
# => math.exp((N / 8) * math.pow(0.1, 2)) = 4 * m_h(2 * N) / delta
# => delta = 4 * m_h(2 * N) * math.exp(-(N / 8) * math.pow(0.1, 2))
# => delta = 4 * (100 + 1) * math.exp(-(100 / 8) * math.pow(0.1, 2))
# for m_h(N) = N + 1 and N = 100
# delta = 356.5287486441765
# so E_out will be at most 0.1 bigger than E_in with probability -355.5287486441765 :(
# or with probability 356.5287486441765 E_out will be with 0.1 of E_in :(

class vc_bound:
    def order_bound(self, N, d_vc):
        return (N ** d_vc) + 1
    
    def weight_bound(self, N, d_vc):
        return ((math.e * N) / d_vc) ** d_vc
    
    def fixed_hoeffding_bound(self, N, delta):
        return np.sqrt((1 / (2 * N)) * np.log(2 / delta))

    def finite_hoeffding_bound(self, N, delta, M):
        return np.sqrt((1 / (2 * N)) * np.log((2 * M) / delta))
    
    def order_confidence_from(self, N, d_vc, epsilon):
        return 4 * self.order_bound(2 * N, d_vc) * math.exp(-(N / 8) * (epsilon ** 2))
    
    def order_tolerance_from(self, N, d_vc, delta):
        return math.sqrt((8 /  N) * math.log((4 / delta) * self.order_bound(2 * N, d_vc)))
    
    #i did it this way because I wanted a "partial apply" so that the function can be used with vc_sample_size and only need N
    #don't know why I try these kind of things in python but may need to adjust later
    
    def order_vc_complexity(self, epsilon, delta, d_vc):
    
        def sample_complexity(N):
            return (8 / (epsilon ** 2)) * math.log((4 / delta) * self.order_bound(2 * N, d_vc))
        
        return sample_complexity
    
    def weight_vc_complexity(self, epsilon, delta, d_vc):
    
        def sample_complexity(N):
            return (8 / (epsilon ** 2)) * math.log((4 / delta) * self.weight_bound(2 * N, d_vc))

        return sample_complexity

    def find_end(self, complexity):
        N = 2 ** 6
        while N < complexity(N):
            N *= 2
        return N
    


    def sample_size_vc(self, complexity):
        start = 2 ** 6
        end = self.find_end(complexity)
        while start <= end:
            mid = (start + end) // 2
            if mid >= complexity(mid):
                end = mid - 1
            else:
                start = mid + 1

        return start

    def finite_model_sample_size(self, M, epsilon, delta):
        return np.round((1 / (2 * (epsilon ** 2))) * math.log((2 * M) / delta), 0)

    def variance_plot(self, N):
        epsilon = 0.1
        p = np.arange(0, 1 + epsilon, epsilon)
        v = p * (1 - p) / N

        plt.plot(p, v, marker = '*')
        plt.xlabel("E_out probability")
        plt.ylabel("Variance of E_out")
        plt.legend()
        plt.show()


    
def test():
    N = 100
    d_vc = 2
    epsilon = 0.1

    bound = vc_bound()
    result = bound.order_tolerance_from(N, d_vc, bound.order_confidence_from(N, d_vc, epsilon))
    print(f"the result is {result}\n")

#test()

def test_plot():
    bound = vc_bound()
    N = 100
    bound.variance_plot(N)

#test_plot()

def test_sample_size_vc():
    d_vc = 5
    epsilon = 0.1
    delta = 0.1
    bound = vc_bound()
    complexity = bound.order_vc_complexity(epsilon, delta, d_vc)
    size = bound.sample_size_vc(complexity)
    print(f"the sample size needed for {(1 - delta) * 100}% confidence that generalization error is within {epsilon * 100}% of E_in is {size}\n")

#test_sample_size_vc()

def test_order_tolerance_from():
    N = 1000
    delta = 0.1
    d_vc = 1
    bound = vc_bound()
    epsilon = bound.order_tolerance_from(N, d_vc, delta)
    print(f"with {(1 - delta) * 100}% confidence with {N} examples we can offer an error deviation of at most {np.round(epsilon, 3)} from E_in")
#test_order_tolerance_from()

def test_fixed_hoeffding_bound():
    N = 1000
    delta = 0.02
    bound = vc_bound()
    epsilon = bound.fixed_hoeffding_bound(N, delta)
    print(f"with {(1 - delta) * 100}% confidence with {N} examples E_out will be at most {np.round(epsilon, 2)} bigger than E_in\n")

#test_fixed_hoeffding_bound()

def test_finite_hoeffding_bound():
    N = 600
    N_test = 100
    N_train = N - N_test
    M = 1000

    delta = 0.05
    bound = vc_bound()
    epsilon_test = bound.fixed_hoeffding_bound(N_test, delta)
    epsilon_train = bound.finite_hoeffding_bound(N_train, delta, M)
    
    print(f"with {(1 - delta) * 100}% confidence with {N_test} examples E_out will be at most {np.round(epsilon_test, 2)} bigger than E_in using reserved Test data to estimate E_out with E_in\n")
    print(f"with {(1 - delta) * 100}% confidence with {N_train} examples E_out will be at most {np.round(epsilon_train, 2)} bigger than E_in using training data to estimate E_out with E_in\n")

#test_finite_hoeffding_bound()

def test_finite_model_sample_size():
    bound = vc_bound()
    delta = 0.03
    epsilon = 0.05
    Ms = np.array([1, 100, 10000, 1000000])

    result = np.array([bound.finite_model_sample_size(M, epsilon, delta) for M in Ms])
    delta_result = np.diff(np.ceil(result))
    print(f"to be assured that E_out is within {epsilon} of E_in\nwith probability {1 - delta} for model complexities: {Ms}\nthe sample sizes required are: {np.ceil(result)}\n")
    print(f"change in sample size required for exponential increase in model complexity is {delta_result}\n")
#test_finite_model_sample_size()

def test_sample_size_vc_v1():
    d_vc = 10
    start = 10 * d_vc
    epsilon = 0.05
    delta = 0.05
    bound = vc_bound()
    complexity = bound.order_vc_complexity(epsilon, delta, d_vc)
    size = bound.sample_size_vc(complexity, start)
    print(f"the sample size needed for {(1 - delta) * 100}% confidence that generalization error is within {epsilon} of E_in is {size}\n")

#test_sample_size_vc_v1()

def test_hw3p123():
    bound = vc_bound()
    epsilon = 0.05
    delta = 0.03
    M = [1, 10, 100]
    N = [bound.finite_model_sample_size(m, epsilon, delta) for m in M]
    print(f"the minimal sample size needed for {M} sized hypothesis with error tolerance {epsilon} and confidence {(1 - delta) * 100}% is {N}\n")

#test_hw3p123()

def test_hw4p1():
    bound = vc_bound()
    epsilon = 0.05
    delta = 0.05
    d_vc = 10
    complexity = bound.order_vc_complexity(epsilon, delta, d_vc)
    N = bound.sample_size_vc(complexity)
    print(f"the minimal sample size needed for hypothesis of vc dimension {d_vc} with error tolerance {epsilon} and confidence {(1 - delta) * 100}% is {N}\n")

test_hw4p1()