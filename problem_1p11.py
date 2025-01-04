import numpy as np
class error_measures:
    def from_cost(cost_matrix, training_data, w, binary_labels):
        X, y = training_data
        num_examples = X.shape[0]
        false, true = binary_labels
        error_matrix = np.zeros((2, 2))
        
        false_y = 0
        true_y = 1
        false_y_prime = 0
        true_y_prime = 1

        y_prime = np.sign(np.dot(X, w))

        for i in range(num_examples):
            if y[i] == false and y_prime[i] == false:
                error_matrix[false_y][false_y_prime] += 1
            elif y[i] == false and y_prime[i] == true:
                error_matrix[false_y][true_y_prime] += 1
            elif y[i] == true and y_prime[i] == false:
                error_matrix[true_y][false_y_prime] += 1
            else:
                error_matrix[true_y][true_y_prime] += 1

        return np.sum(cost_matrix * error_matrix)




