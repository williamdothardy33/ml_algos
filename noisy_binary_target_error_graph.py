import numpy as np
import matplotlib.pyplot as plt
class noisy_binary_target:
    def plot_error_graph(self):
        delta = 0.1
        fig = plt.figure()
        #ax = plt.axes(projection="3d")
        ax = fig.add_subplot(111, projection="3d")
        


        e_out_probabilities = np.arange(0, 1 + delta, delta)
        quiet_probabilities = np.arange(0, 1 + delta, delta)

        m, l = np.meshgrid(e_out_probabilities, quiet_probabilities)

        print(f"m is {m}\n")
        print(f"l is {l}\n")
    
        

        noisy_error = l * m + (1 - m) * (1 - l)
        print(f"noisy error is {noisy_error}\n")

        # ax.set_title("noisy binary target error (P(h != y)) from: out of sample error vs conditional probability of outcome")
        # ax.set_xlabel("P(h != f)")
        # ax.set_ylabel("P(y == f)")


        #surf = ax.plot_surface(m, l, noisy_error, cmap="viridis")

        # plt.figure()
        # for mu in e_out_probabilities:
        #     l_cross = l[:, e_out_probabilities == mu][:, 0]
        #     noisy_error_cross = noisy_error[:, e_out_probabilities == mu][:, 0]
        #     plt.plot(l_cross, noisy_error_cross, label=f"mu={mu}")

        #check_noisy_error_cross = 0.2 * quiet_probabilities + (1 - 0.2) * (1 - quiet_probabilities)
        #print(f"noisy error cross {noisy_error_cross}\n")
        #print(f"check l_cross is {check_noisy_error_cross}\n")
        #print(f"they are equal: {noisy_error_cross == check_noisy_error_cross}\n")

        #l_cross_1 = l[:, e_out_probabilities == 0.4][:, 0]
    
        #noisy_error_cross_1 = noisy_error[:, e_out_probabilities == 0.4][:, 0]

        #plt.figure()
        #plt.plot(l_cross, noisy_error_cross)
        #plt.plot(l_cross_1, noisy_error_cross_1)


        #plt.legend()
        #plt.show()


def test():
    noisy_binary_target().plot_error_graph()

test()




