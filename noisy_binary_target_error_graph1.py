import numpy as np
import matplotlib.pyplot as plt
class noisy_binary_target:
    def plot_error_graph(self):
        delta = 0.1

        e_out_probabilities = np.arange(0, 1 + delta, delta)
        quiet_probabilities = np.arange(0, 1 + delta, delta)

        m, l = np.meshgrid(e_out_probabilities, quiet_probabilities)
    
        noisy_error = l * m + (1 - m) * (1 - l)

        fig = plt.figure(figsize=(15,5))

        ax1 = fig.add_subplot(1,3,1, projection='3d')
        ax1.set_title("noisy binary target error (P(h != y))\nfrom: out of sample error vs\nconditional probability of outcome")
        ax1.set_xlabel("P(h != f)")
        ax1.set_ylabel("P(y == f)")
        ax1.plot_surface(m, l, noisy_error, cmap="viridis")

        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)
        for lmda in quiet_probabilities:
            m_cross = m[quiet_probabilities == lmda, :][0]
            print(f"m_cross is {m_cross}\n")
            noisy_error_cross = noisy_error[quiet_probabilities == lmda, :][0]
            print(f"noisy cross is {noisy_error_cross}\n")
            if lmda <= 0.5:
                ax2.plot(m_cross, noisy_error_cross, label=f"lamda={np.round(lmda, 2)}")
            if lmda >= 0.5:
                ax3.plot(m_cross, noisy_error_cross, label=f"lamda={np.round(lmda, 2)}")

        ax2.set_title("(P(h != y)) vs P(h != f)")
        ax2.set_xlabel("P(h != f)")
        ax2.set_ylabel("P(h != y)")
        ax2.legend()


        ax3.set_title("(P(h != y)) vs P(h != f)")
        ax3.set_xlabel("P(h != f)")
        ax3.set_ylabel("P(h != y)")
        ax3.legend()

        plt.tight_layout()
        plt.show()


def test():
    noisy_binary_target().plot_error_graph()

test()




