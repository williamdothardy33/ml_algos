import numpy as np
import matplotlib.pyplot as plt

#biased_coin cherry picks the samples from the different sampling distributions, so for each sample head count from the sammpling distributions
#it will pick the sampling distribution with the smallest count

class coin_sim:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.distributions = None
        self.first_idx = 0
        self.rand_idx = 1
        self.biased_idx = 2

    def pick_coins(self, num_coins, num_flips, diagnostic = False):

        first_coin = 0
        random_coin = self.rng.choice(num_coins)
        head_freqs = self.rng.binomial(num_flips, 0.5, num_coins)

        biased_coin = np.argmin(head_freqs)

        if diagnostic:
            print('*****************************************************************************\n')
            print(f"first coin flipped {head_freqs[first_coin]} heads\n")
            print(f"random coin at index {random_coin} flipped {head_freqs[random_coin]} heads\n")
            print(f"biased coin at index {biased_coin} flipped {head_freqs[biased_coin]} heads\n")
            print('*****************************************************************************\n')

        return head_freqs[first_coin] / num_flips, head_freqs[random_coin] / num_flips, head_freqs[biased_coin] / num_flips

    def sim_picks(self, num_iter, num_coins, num_flips):
        distributions = np.empty((3, num_iter))
        for i in range(num_iter):
            for k, coin_avg in enumerate(self.pick_coins(num_coins, num_flips)):
                distributions[k][i] = coin_avg

        self.distributions = distributions

    def graph_analysis(self, num_flips):
        if self.distributions is None:
            print("Warning, no picked distributions found. Please run sim_picks.")
            return

        num_coins = 3
        num_examples = self.distributions.shape[1]

        colors = ['red', 'green', 'blue']
        labels = ['first coin', 'random coin', 'biased coin']

        _, axs = plt.subplots(2, num_coins, tight_layout=True)

        for i in range(num_coins):
            axs[0, i].hist(self.distributions[i], bins=num_flips, color=colors[i])
            axs[0, i].set_title(labels[i])

        delta = 1 / num_flips

        e_x = np.arange(0, 1 + delta, delta)
        e_y = 2 * np.exp(-2 * np.pow(e_x, 2) * num_flips)

        mean_abs_diff = np.abs(self.distributions - 0.5)
        P_y = np.empty((num_coins, e_x.shape[0]))

        for n in range(num_coins):
            for j, x in enumerate(e_x):
                P_y[n][j] = np.sum(mean_abs_diff[n] > x) / num_examples

        for k in range(num_coins):
            axs[1, k].plot(e_x, e_y, marker='o')
            axs[1, k].plot(e_x, P_y[k])

        plt.show()
    
def test_pick_coins():
    num_coins = 1000
    num_flips = 10
    coin_sim().pick_coins(num_coins, num_flips, True)

#test_pick_coins()

def test_sim_picks():
    num_iter = 20
    num_coins = 1000
    num_flips = 10
    sim = coin_sim()
    sim.sim_picks(num_iter, num_coins, num_flips)
    print(f"the distribution of picked coins is\n{sim.distributions}\n")

#test_sim_picks()

def test_graph_analysis():
    num_iter = 10000
    num_coins = 1000
    num_flips = 10
    sim = coin_sim()
    sim.sim_picks(num_iter, num_coins, num_flips)
    sim.graph_analysis(num_flips)

#test_graph_analysis()

def test_averages():
    num_iter = 100000
    num_coins = 1000
    num_flips = 10
    sim = coin_sim()
    sim.sim_picks(num_iter, num_coins, num_flips)
    print(f"the average value of v_min is {np.average(sim.distributions[sim.biased_idx])}\n")
    print(f"the average value of v_first is {np.average(sim.distributions[sim.first_idx])}\n")
    print(f"the average value of v_rand is {np.average(sim.distributions[sim.rand_idx])}\n")


#test_averages()
