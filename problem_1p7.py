import numpy as np
import matplotlib.pyplot as plt
import math
class hoeffding:
    def coin_max_distribution(self, num_coins, num_flips, mu):
        #num_coins = 2
        #num_flips = 6
        #mu = 0.5
        average = num_flips * mu
        num_heads = np.arange(0, num_flips + 1, 1)
        p_num_heads = np.array([math.comb(num_flips, k) * (mu ** k) * (1 - mu) ** (num_flips - k) for k in num_heads])
        deviations = np.abs(num_heads - average)
        max_deviation = np.max(deviations)
        deviation_scale = np.arange(0, max_deviation + 1)
        p_deviation_scale = np.array([np.sum(p_num_heads[deviations <= deviation]) for deviation in deviation_scale])
        p_max = 1 - p_deviation_scale ** num_coins
        p_hoeffding = 2 * np.exp(-2 * ((deviation_scale / num_flips) ** 2) * num_flips)
        return deviation_scale / num_flips, p_max, p_hoeffding

    def graph_analysis(self):
        base = 2
        num_coins = np.array([base ** i for i in range(1, 21, 2)])
        num_flips = 6
        mu = 0.5

        for num_coin in num_coins:
            scale, p_max, p_hoeffding = self.coin_max_distribution(num_coin, num_flips, mu)
            plt.plot(scale, p_max, marker='*', label=f"{num_coin} coins")

        plt.plot(scale, p_hoeffding, marker='o', color='blue')
        plt.title(f"Probability that deviation exceeds threshold\nnum flips={num_flips}\ne_out(max_coin)={mu}\n")
        plt.xlabel("threshold e")
        plt.ylabel("P(|e_in(max coin) - e_out(max coin)| > e)")
        plt.legend()
        plt.show()






        




        
def test():
    h = hoeffding()
    h.graph_analysis()

test()