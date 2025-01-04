import numpy as np
class bias_variance_sim:
    def __init__(self):
        self.rng = np.random.default_rng()
    
    def learning_algo_affine(self, training_data):
        x, y = training_data
        a = (y[0] - y[1]) / (x[0] - x[1])
        b = ((y[1] * x[0]) - (y[0] * x[1])) / (x[0] - x[1])

        return (a, b)
    
    def learning_algo_linear(self, training_data):
        x, y = training_data
        a = (y[0] * x[0] + y[1] * x[1]) / ((x[0] ** 2) + (x[1] ** 2)) #I think a is just the projection of y onto x. I guess this makes sense as that projection would be the best "representative" of y if we only have span(x) to get as close as possible to y (I originally did this with calculus by finding a that minimized average square deviation)

        return a
    
    def learning_algo_constant(self, training_data):
        _, y = training_data
        b = (y[0] + y[1]) / 2

        return b
    
    def mse(self, predicted, actual):
        return np.average(np.pow(predicted - actual, 2))
    
    def analysis(self):
        test_in = self.rng.uniform(-1.0, 1.0, 10000)
        test_out = np.sin(test_in * np.pi)

        datasets_in = self.rng.uniform(-1, 1, (1000, 2))
        datasets_out = np.sin(datasets_in * np.pi)

        learned_affine = np.empty((1000, 2))
        learned_linear = np.empty((1000, 1))
        learned_constant = np.empty((1000, 1))

        for i, dataset_in in enumerate(datasets_in):
            training_data = (dataset_in, datasets_out[i])
            learned_affine[i][0], learned_affine[i][1] = self.learning_algo_affine(training_data)
            learned_linear[i][0] = self.learning_algo_linear(training_data)
            learned_constant[i][0] = self.learning_algo_constant(training_data)

        g_bar_affine = np.mean(learned_affine, axis=0)
        g_bar_linear = np.mean(learned_linear, axis=0)
        g_bar_constant = np.mean(learned_constant, axis=0)

        #var_affine_x = np.var(learned_affine, axis=0, mean=g_bar_affine)
        #cov_affine_x = np.cov((learned_affine - np.ones((learned_affine.shape[0], 2)) * g_bar_affine), rowvar=False, bias=True)
        cov_affine_x = np.cov((learned_affine - np.ones((learned_affine.shape[0], 2)) * g_bar_affine), rowvar=False)
        avg_sq_dev_affine = np.mean((test_in ** 2) * cov_affine_x[0][0] + 2 * cov_affine_x[0][1] * test_in + np.ones(test_in.size) * cov_affine_x[1][1])
        bias_affine = np.mean(((g_bar_affine[0] * test_in + np.ones(test_in.size) * g_bar_affine[1]) - test_out) ** 2)
        #var_affine_x_v1 = np.mean((learned_affine - np.ones((learned_affine.shape[0], 2)) * g_bar_affine) ** 2, axis=0)
        var_linear_x = np.var(learned_linear, axis=0, mean=g_bar_linear, ddof=1)
        avg_sq_dev_linear = np.mean((test_in ** 2) * var_linear_x)
        bias_linear = np.mean(((g_bar_linear * test_in) - test_out) ** 2)
        var_constant_x = np.var(learned_constant, axis=0, mean=g_bar_constant, ddof=1)
        avg_sq_dev_constant = var_constant_x
        bias_constant = np.mean(((np.ones(test_out.size) * g_bar_constant) - test_out) ** 2)

        #print(f"var_affine_x is {var_affine_x}\n")
        #print(f"var_affine long way is {var_affine_x_v1}\n")
        #print(f"cov matrix computed is {cov_affine_x}\n")
        print(f"the best hypothesis in the set of affine models is {g_bar_affine}\n")
        print(f"average square error in predictions for affine model picked from the 'best one' available is {avg_sq_dev_affine}\n")
        print(f"the average irreducible square error in predictions between the affine model picked and the target is {bias_affine}\n")

        print(f"the best hypothesis in the set of linear models is {g_bar_linear}\n")
        print(f"average square error in predictions for linear model picked from the 'best one' available is {avg_sq_dev_linear}\n")
        print(f"the average irreducible square error in predictions between the linear model picked and the target is {bias_linear}\n")

        print(f"the best hypothesis in the set of constant models is {g_bar_constant}\n")
        print(f"average square error in predictions for constant model picked from the 'best one' available is {avg_sq_dev_constant}\n")
        print(f"the average irreducible square error in predictions between the model picked and the target is {bias_constant}\n")



def test():
    sim = bias_variance_sim()
    sim.analysis()

test()

        