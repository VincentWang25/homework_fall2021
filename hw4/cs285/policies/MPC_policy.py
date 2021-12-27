import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            # TODO(Q1) uniformly sample trajectories and return an array of
            # dimensions (num_sequences, horizon, self.ac_dim) in the range
            # [self.low, self.high]
            random_action_sequences = np.random.rand(num_sequences, horizon, self.ac_dim) * (self.high - self.low) + self.low
            return random_action_sequences
        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf
            act_mu = None
            act_var = None
            for i in range(self.cem_iterations):
                if i == 0:
                    act = np.random.rand(num_sequences, horizon, self.ac_dim) * (self.high - self.low) + self.low
                else:
                    act = np.zeros((num_sequences, horizon, self.ac_dim))
                    for h in range(horizon):
                        act[:, h, :] = np.random.multivariate_normal(act_mu[h], act_var[h], num_sequences)                    
                rewards = self.evaluate_candidate_sequences(act, obs)
                elite_idx = np.argpartition(rewards, -self.cem_num_elites)[-self.cem_num_elites:]
                elite_act = act[elite_idx]
                
                act_mu_tmp = np.mean(elite_act, axis=0)
                act_var_tmp = np.zeros((horizon, self.ac_dim, self.ac_dim))
                for h in range(horizon):
                    act_var_tmp[h, :, :] = np.cov(elite_act[:, h, :], rowvar=False)                
                
                if i == 0:
                    act_mu = act_mu_tmp
                    act_var = act_var_tmp
                else:
                    act_mu = self.cem_alpha * act_mu_tmp + (1 - self.cem_alpha) * act_mu
                    act_var = self.cem_alpha * act_var_tmp + (1 - self.cem_alpha) * act_var
                
                    
                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance

            # TODO(Q5): Set `cem_action` to the appropriate action sequence chosen by CEM.
            # The shape should be (horizon, self.ac_dim)
            cem_action = act_mu

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        sum_of_rewards = np.zeros(self.N)
        for model in self.dyn_models:
            sum_of_rewards += self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            
        return sum_of_rewards / len(self.dyn_models)

    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            index = np.argmax(predicted_rewards)
            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = candidate_action_sequences[index]  # TODO (Q2)
            action_to_take = best_action_sequence[0]  # TODO (Q2)
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        # TODO (Q2)
        old_states = np.repeat(obs[None], self.N, 0)
        sum_of_rewards = np.zeros(self.N)  
        for i in range(self.horizon):            
            acts = candidate_action_sequences[:,i]
            new_states = model.get_prediction(old_states, acts, self.data_statistics)
            sum_of_rewards += self.env.get_reward(new_states, acts)[0]
            old_states = new_states
            
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)` at each step.
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.
        return sum_of_rewards
