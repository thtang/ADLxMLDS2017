from agent_dir.agent import Agent
import tensorflow as tf 
import numpy as np 
import random
from collections import deque
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Hyper Parameters:
FRAME_PER_ACTION = 1

GAMMA = 0.95 # decay rate of past observations

OBSERVE = 50000. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon

FINAL_EPSILON = 0.1#0.001 # final value of epsilon
INITIAL_EPSILON = 1.0#0.01 # starting value of epsilon

REPLAY_MEMORY = 1000000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 10000
NUM_EPISODES = 100000
MAX_NUM_STEPS = 10000

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        
        tf.reset_default_graph()
        self.env = env
        self.args = args
        # init replay memory
        self.replayMemory = deque()
        # init some parameters
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = env.action_space.n
        # init Q network
        self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

        # init Target Q Network
        self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()

        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)

        if args.test_dqn:
            #you can load your model here
            print('testing------- loading trained model-------')
            # model_file = tf.train.latest_checkpoint("./test_model")
            model_file = "./tf_DQN-15240000"
            self.saver.restore(self.session, model_file)
            print("Model restored.")

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def createQNetwork(self):
        # network weights
        W_conv1 = self.weight_variable([8,8,4,32])
        b_conv1 = self.bias_variable([32])

        W_conv2 = self.weight_variable([4,4,32,64])
        b_conv2 = self.bias_variable([64])

        W_conv3 = self.weight_variable([3,3,64,64])
        b_conv3 = self.bias_variable([64])

        W_fc1 = self.weight_variable([3136,512])
        b_fc1 = self.bias_variable([512])

        W_fc2 = self.weight_variable([512,self.actions])
        b_fc2 = self.bias_variable([self.actions])

        # input layer

        stateInput = tf.placeholder("float",[None,84,84,4])

        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
        #h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
        h_conv3_shape = h_conv3.get_shape().as_list()
        print ("dimension:",h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3])
        h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

        # Q Value layer
        QValue = tf.matmul(h_fc1,W_fc2) + b_fc2

        return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

    def copyTargetQNetwork(self):
        self.session.run(self.copyTargetQNetworkOperation)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float",[None,self.actions])
        self.yInput = tf.placeholder("float", [None]) 
        Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
        self.trainStep = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)

    def trainQNetwork(self):

        
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step 2: calculate y 
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        self.trainStep.run(feed_dict={
            self.yInput : y_batch,
            self.actionInput : action_batch,
            self.stateInput : state_batch
            })

        # save network every 100000 iteration
        if self.timeStep % 10000 == 0:
            self.saver.save(self.session, './save_model/tf_DQN', global_step = self.timeStep)

        if self.timeStep % UPDATE_TIME == 0:
            self.copyTargetQNetwork()

    def setPerception(self,current_state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.actions)
        one_hot_action[action] = 1
        self.replayMemory.append((current_state,one_hot_action,reward,next_state,done))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if len(self.replayMemory) > BATCH_SIZE:
            # skip frame
            if self.timeStep % 4 ==0:
                # Train the network
                self.trainQNetwork()
        self.timeStep += 1

    def make_action(self,observation, test=True):
        observation = observation.reshape((1,84,84,4))
        QValue = self.QValue.eval(feed_dict={self.stateInput:observation})[0]
        
        if random.random() <= self.epsilon and not test:
            action = random.randrange(self.actions)
        else:
            action = np.argmax(QValue)
        
        if test and random.random()>0.01:
            action = np.argmax(QValue)
        elif test:
            action = random.randrange(self.actions)

        if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
        return action
    def train(self):
        """
        Implement your training algorithm here
        """
        # self.env
        # action0 = [1,0,0,0]  # do nothing
        # observation0, reward0, terminal, _ = self.env.step(np.argmax(action0))
        print("environment output shape:",self.env.reset().shape)
        learning_history = []
        for e in range(NUM_EPISODES):
            observation = self.env.reset() # (84,84,4)
            step_count = 0
            total_reward = 0
            current_state = observation

            for s in range(MAX_NUM_STEPS):
                action = self.make_action(current_state, test=False)
                next_state,reward, done, _ = self.env.step(action)
                
                self.setPerception(current_state,action,reward,next_state, done)
                current_state = next_state

                total_reward += reward
                step_count +=1 

                if done == True:
                    print("episode:", e, " step_count:",step_count," reward:",total_reward," total time steps:",self.timeStep)
                    learning_history.append((e,step_count,total_reward,self.timeStep))
                    break
            if e % 1000 ==0:
                np.save("dqn_learning_history.npy", np.array(learning_history))

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(self,x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")