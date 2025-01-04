import numpy as np
import matplotlib.pyplot as plt

class problem1p2cont:
    def draw_lines(self, w_one, w_two):
        fig, ax = plt.subplots()

        a_1 = -(w_one[1] / w_one[2])
        b_1 = -(w_one[0] / w_one[2])

        a_2 = -(w_two[1] / w_two[2])
        b_2 = -(w_two[0] / w_two[2])


        x = np.arange(-10,11, 1)
        y_1 = a_1 * x + b_1
        y_2 = a_2 * x + b_2
        
        ax.plot(x, y_1, label=f"{w_one}")
        ax.plot(x, y_2, label=f"{w_two}")
        plt.legend()
        plt.show()

def test():
    w_one = np.array([1,2,3])
    w_two = -np.array([1,2,3])
    problem1p2cont().draw_lines(w_one, w_two)

test()
