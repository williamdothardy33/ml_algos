import numpy as np
#some ideas that I came up with and got from gemini were an "exploratory push" where we can take a percentage of times steps in the beginning
#and use them solely for exploratory moves. we can also do a "variable" explore rate where in the beginning we start off with a higher explore rate
#and then decrease it down to the desired argument level (we can experiment with how to throttle the rate down)
class k_bandit_sim:
    def __init__(self, seed = None):
        if not seed:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

        self.action_value_map = None

    def gen_actions(self, num_actions = 10):
        self.action_value_map = self.rng.standard_normal(num_actions)
        return np.arange(num_actions)
        
    
    def reward_t(self, action):
        if self.action_value_map is None:
            print(f'please generate actions with gen_actions before trying to retrieve a reward\n')
            return
        if action >= self.action_value_map.size:
            print(f'invalid action')
            return
        
        return self.rng.normal(self.action_value_map[action], 1)
    
    def epsilon_greedy_learning_algo(self, actions, explore_rate, max_time_step):
        action_freq = np.zeros(actions.size)
        action_reward_sum = np.zeros(actions.size)
        action_value_estimate = np.zeros(actions.size)

        for t in range(max_time_step):
            should_explore = self.rng.random()
            if should_explore < explore_rate or t == 0:
                next_action = self.rng.choice(actions)
                action_freq[next_action] += 1
                action_reward_sum[next_action] += self.reward_t(next_action)
                action_value_estimate[next_action] = action_reward_sum[next_action] / action_freq[next_action]
            else:
                next_action = np.argmax(action_value_estimate)
                action_freq[next_action] += 1
                action_reward_sum[next_action] += self.reward_t(next_action)
                action_value_estimate[next_action] = action_reward_sum[next_action] / action_freq[next_action]

        print(f'the probability for each action taken for this run is {action_freq / max_time_step}\n')

        return action_value_estimate

    def linear_epsilon_greedy_learning_algo(self, actions, explore_rate, explore_push_period, max_time_step):
        
        action_freq = np.zeros(actions.size)
        action_reward_sum = np.zeros(actions.size)
        action_value_estimate = np.zeros(actions.size)

        for t in range(max_time_step):
            should_explore = self.rng.random()
            if should_explore < explore_rate or t == 0:
                next_action = self.rng.choice(actions)
                action_freq[next_action] += 1
                action_reward_sum[next_action] += self.reward_t(next_action)
                action_value_estimate[next_action] = action_reward_sum[next_action] / action_freq[next_action]
            else:
                next_action = np.argmax(action_value_estimate)
                action_freq[next_action] += 1
                action_reward_sum[next_action] += self.reward_t(next_action)
                action_value_estimate[next_action] = action_reward_sum[next_action] / action_freq[next_action]

        print(f'the probability for each action taken for this run is {action_freq / max_time_step}\n')

        return action_value_estimate
    
def test_egla():
    sim = k_bandit_sim()
    actions = sim.gen_actions()
    explore_rate = 0.1
    max_time_step = 1000
    learned_values = sim.epsilon_greedy_learning_algo(actions, explore_rate, max_time_step)

    print(f'the target action value map is {sim.action_value_map}\n')
    print(f'the estimated action value map is {learned_values}\n')

test_egla()


    