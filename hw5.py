import numpy as np
class hw5:
    def non_projected_error(self, noise_variance, feature_length, num_examples):
        return noise_variance * (1 - ((feature_length + 1) / num_examples))
    
    def sample_size_from(self, noise_variance, feature_length, expected_ein):
        return np.round((feature_length + 1) / (1 - (expected_ein / noise_variance)), 0)


    def my_error(self, loc):
        u, v = loc
        return (u * (np.e ** v) - 2 * v * (np.e ** -u)) ** 2

    
    def my_error_descent_bearing(self, loc):
        u, v = loc
        delta_u = 2 * ((u * (np.e ** v)) - 2 * v * (np.e ** (-u))) * ((np.e ** v) + 2 * v * (np.e ** (-u)))
        delta_v = 2 * ((u * (np.e ** v)) - 2 * v * (np.e ** (-u))) * (u * (np.e ** v) - 2 * np.e ** (-u))
        return np.array([delta_u, delta_v])

    def my_error_bearing_u_descent(self, loc):
        u, v = loc
        delta_u = 2 * ((u * (np.e ** v)) - 2 * v * (np.e ** (-u))) * ((np.e ** v) + 2 * v * (np.e ** (-u)))
        return np.array([delta_u, v])

    def my_error_bearing_v_descent(self, loc):
        u, v = loc
        delta_v = 2 * ((u * (np.e ** v)) - 2 * v * (np.e ** (-u))) * (u * (np.e ** v) - 2 * np.e ** (-u))
        return np.array([u, delta_v])


    def gradient_descent(self, start, descent_rate, max_iter, min_error):
        loc = start.copy()
        current_error = self.my_error(loc)
        
        for i in range(max_iter):
            if current_error < min_error:
                break

            next_bearing = -self.my_error_descent_bearing(loc)
            loc += descent_rate * next_bearing

            current_error = self.my_error(loc)

        return i, loc

    def coordinate_descent(self, start, descent_rate, max_iter, min_error):
        loc = start.copy()
        current_error = self.my_error(loc)
        
        for i in range(max_iter):
            if current_error < min_error:
                break

            next_bearing_u = -self.my_error_bearing_u_descent(loc)
            loc += descent_rate * next_bearing_u
            current_error = self.my_error(loc)

            next_bearing_v = -self.my_error_bearing_v_descent(loc)
            loc += descent_rate * next_bearing_v
            current_error = self.my_error(loc)

        return i, loc

def test_hw5_sample():
    hw = hw5()

    noise_variance = 0.1 ** 2
    feature_length = 8
    min_expected_ein = 0.008
    print(f'the minimal sample sized needed for in sample error at least {min_expected_ein} on average is {hw.sample_size_from(noise_variance, feature_length, min_expected_ein)}\n')

#test_hw5_sample()

def test_gradient_descent():
    hw = hw5()
    min_error = 10 ** (-14)
    start = np.array([1.0, 1.0])
    descent_rate = 0.1
    max_iter = 20
    num_iter, loc = hw.gradient_descent(start, descent_rate, max_iter, min_error)
    print(f'it took {num_iter} iterations to arrive at {loc} on the error surface with error {hw.my_error(loc)}')

#test_gradient_descent()

def test_coordinate_descent():
    hw = hw5()
    min_error = 10 ** (-14)
    start = np.array([1.0, 1.0])
    descent_rate = 0.1
    max_iter = 15
    num_iter, loc = hw.coordinate_descent(start, descent_rate, max_iter, min_error)
    print(f'it took {num_iter} iterations to arrive at {loc} on the error surface with error {hw.my_error(loc)}')

test_coordinate_descent()