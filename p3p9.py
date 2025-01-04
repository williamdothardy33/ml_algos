import numpy as np
import matplotlib.pyplot as plt
class pointwise_error:
    def analysis(self):
        y = 1
        s = np.arange(-10, 11, 1)
        e_class = np.array([1 if y != np.sign(s_i) else 0 for s_i in s])
        e_log = (1 / np.log(2)) * np.log((1 + np.exp(-y * s)))
        e_square = (y - s) ** 2

        plt.plot(s, e_class, label="classification error")
        plt.plot(s, e_log, label="bit error")
        plt.plot(s, e_square, label="square error")
        plt.legend()
        plt.show()

def test_error_plot():
    error = pointwise_error()
    error.analysis()

test_error_plot()


