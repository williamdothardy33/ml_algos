import math
import numpy as np
import matplotlib.pyplot as plt
class growth_function_bounds:
    def order_bound(self, N, d_vc):
        return N ** d_vc + 1
    
    def weight_bound(self, N, d_vc):
        return ((math.e * N) / d_vc) ** d_vc

    def generalization_bound(self, N, upper_bound, delta):
        return 4 * np.sqrt((1 / N) * (np.log(2) + np.log(upper_bound) + np.log(1 / delta)))
    
    def plot(self):
        max_complexity = 5
        d_vcs = np.array([d_vc for d_vc in range(1, max_complexity + 1, 2)])
        Ns = np.arange(1, 20, 2)

        d_vc_bounds_1 = np.empty((d_vcs.size, Ns.size))
        d_vc_bounds_2 = np.empty((d_vcs.size, Ns.size))

        for u, d_vc in enumerate(d_vcs):
            for v, N in enumerate(Ns):
                d_vc_bounds_1[u][v] = self.order_bound(N, d_vc)
                d_vc_bounds_2[u][v] = self.weight_bound(N, d_vc)



        for i, d_vc in enumerate(d_vcs):
            plt.plot(Ns, np.log(d_vc_bounds_1[i]), label=f"bound 1 d_vc={d_vc}")
            plt.plot(Ns, np.log(d_vc_bounds_2[i]), label=f"bound 2 d_vc={d_vc}")

        plt.xlabel("number of examples")
        plt.ylabel("upper  bound of dichotomies for given vc dimension")

        plt.legend()
        plt.show()

def test():
    bounds = growth_function_bounds()
    bounds.plot()
#test()

def test_generalization_bound():
    bounds = growth_function_bounds()
    d_vc = 1
    delta = 0.1
    num_examples = [100, 10000]
    #print(f"{bounds.generalization_bound(100, float(bounds.order_bound(100, d_vc)), delta)}\n")
    #print(f"{bounds.generalization_bound(10000, float(bounds.order_bound(10000, d_vc)), delta)}\n")
    #print(f"{bounds.weight_bound(100, d_vc)}\n")
    #print(f"{bounds.weight_bound(10000, d_vc)}\n")



    moe_1 = [bounds.generalization_bound(n, float(bounds.order_bound(n, d_vc)), delta) for n in num_examples]
    moe_2 = [bounds.generalization_bound(n, float(bounds.weight_bound(n, d_vc)), delta) for n in num_examples]


    print(f"for num_examples={num_examples} and first bound on growth function the E_out is no more than {np.round(moe_1, 5)} from E_in with confidence {(1 - delta) * 100}%\n")
    print(f"for num_examples={num_examples} and second bound on growth function the E_out is no more than {np.round(moe_2, 5)} from E_in with confidence {(1 - delta) * 100}%\n")

    print(bounds.weight_bound(811032.0, 10))
    print(np.round(bounds.generalization_bound(811032.0, bounds.weight_bound(811032.0, 10), 0.05), 2))
test_generalization_bound()
    






