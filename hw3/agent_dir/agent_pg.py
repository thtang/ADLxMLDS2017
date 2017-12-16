from agent_dir.agent import Agent
import scipy
import numpy as np
import os.path
import tensorflow as tf



def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

OBSERVATIONS_SIZE = 6400

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        
        tf.reset_default_graph()
        ##################
        # YOUR CODE HERE #
        ##################
        self.learning_rate = 0.0005
        hidden_layer_size = 200
        checkpoints_dir = "save_model_pg"
        self.env = env
        self.batch_size_episodes = 1
        self.checkpoint_every_n_episodes = 10

        self.sess = tf.InteractiveSession()

        self.observations = tf.placeholder(tf.float32, [None, OBSERVATIONS_SIZE])
        # +1 for up, -1 for down
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(
            tf.float32, [None, 1], name='advantage')

        h = tf.layers.dense(
            self.observations,
            units=hidden_layer_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.up_probability = tf.layers.dense(
            h,
            units=1,
            activation=tf.sigmoid,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.loss = tf.losses.log_loss(
            labels=self.sampled_actions,
            predictions=self.up_probability,
            weights=self.advantage)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(checkpoints_dir,
                                            'policy_network.ckpt')
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            # model_file = tf.train.latest_checkpoint("./save_model_pg")
            model_file = "./policy_network.ckpt"
            self.saver.restore(self.sess, model_file)
            print("Model restored.")

    def load_checkpoint(self):
        print("Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

    def forward_pass(self, observations):
        up_probability = self.sess.run(
            self.up_probability,
            feed_dict={self.observations: observations.reshape([1, -1])})
        return up_probability

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.observation_memory = []


    def train(self):
        UP_ACTION = 2
        DOWN_ACTION = 3
        # Mapping from action values to outputs from the policy network
        action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}
        print(self.env.reset().shape)
        episode_n = 1
        total_time_step = 1
        batch_state_action_reward_tuples = []
        smoothed_reward = None
        learning_history = []
        while True:
            print("starting episode",episode_n)

            episode_done = False
            episode_reward_sum = 0
            round_n = 1
            last_observation = self.env.reset()
            last_observation = prepro(last_observation)
            action = self.env.action_space.sample()
            observation, _, _, _ = self.env.step(action)
            observation = prepro(observation)
            n_steps = 1
            discount_factor = 0.99
            while not episode_done:
                observation_delta = observation - last_observation
                last_observation = observation
                up_probability = self.forward_pass(observation_delta)[0]
                if np.random.uniform() < up_probability:
                    action = UP_ACTION
                else:
                    action = DOWN_ACTION

                observation, reward, episode_done, info = self.env.step(action)
                observation = prepro(observation)
                episode_reward_sum += reward
                n_steps += 1
                total_time_step +=1
                tup = (observation_delta, action_dict[action], reward)
                batch_state_action_reward_tuples.append(tup)

                if reward == -1:
                    print("Round %d: %d time steps; lost..." % (round_n, n_steps))
                elif reward == +1:
                    print("Round %d: %d time steps; won!" % (round_n, n_steps))
                if reward != 0:
                    round_n += 1
                    n_steps = 0
            print("Episode %d finished after %d rounds" % (episode_n, round_n))
            # exponentially smoothed version of reward
            if smoothed_reward is None:
                smoothed_reward = episode_reward_sum
            else:
                smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum * 0.01
            print("Reward total was %.3f; discounted moving average of reward is %.3f" \
                % (episode_reward_sum, smoothed_reward))

            learning_history.append((episode_n,total_time_step,episode_reward_sum))

            if episode_n % self.batch_size_episodes == 0:
                states, actions, rewards = zip(*batch_state_action_reward_tuples)
                rewards = self.discount_rewards(rewards, discount_factor)
                rewards -= np.mean(rewards)
                rewards /= np.std(rewards)
                batch_state_action_reward_tuples = list(zip(states, actions, rewards))
                self.input_train(batch_state_action_reward_tuples)
                batch_state_action_reward_tuples = []
            if episode_n % self.checkpoint_every_n_episodes == 0:
                self.save_checkpoint()
                np.save("pg_learning_history.npy",learning_history)
            episode_n += 1

    def discount_rewards(self, rewards, discount_factor):
        discounted_rewards = np.zeros_like(rewards)
        for t in range(len(rewards)):
            discounted_reward_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                discounted_reward_sum += rewards[k] * discount
                discount *= discount_factor
                if rewards[k] != 0:
                    # Don't count rewards from subsequent rounds
                    break
            discounted_rewards[t] = discounted_reward_sum
        return discounted_rewards

    def input_train(self, state_action_reward_tuples):
        print("Training with %d (state, action, reward) tuples" %
              len(state_action_reward_tuples))

        states, actions, rewards = zip(*state_action_reward_tuples)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)

        feed_dict = {
            self.observations: states,
            self.sampled_actions: actions,
            self.advantage: rewards
        }
        self.sess.run(self.train_op, feed_dict)

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        UP_ACTION = 2
        DOWN_ACTION = 3
        if self.observation_memory == []:
            init_observation = prepro(observation)
            action = self.env.get_random_action()
            second_observation, _, _, _ = self.env.step(action)
            second_observation = prepro(second_observation)
            observation_delta = second_observation - init_observation
            self.observation_memory = second_observation
            up_probability = self.forward_pass(observation_delta)[0]
            if up_probability > 0.5:
                action = UP_ACTION
            else:
                action = DOWN_ACTION
        else:
            observation = prepro(observation)
            observation_delta = observation - self.observation_memory
            self.observation_memory = observation
            up_probability = self.forward_pass(observation_delta)[0]
            if up_probability > 0.5:
                action = UP_ACTION
            else:
                action = DOWN_ACTION
        # action = self.env.get_random_action()
        return action