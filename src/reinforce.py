import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
from keras import initializers
from keras import losses
from keras import Sequential
from keras.layers import Dense

#import torch
#import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr=1e-4):
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        self.model = model
        self.model.compile(loss=losses.categorical_crossentropy, 
                           optimizer=keras.optimizers.Adam(learning_rate=lr))


    def train(self, env, gamma=0.99):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        states, actions, rewards = self.generate_episode(env)
        
        T = len(states)
        G = [] # Gain list of every step
        
        r = np.array(rewards) # (T,)
        G = np.zeros_like(r) # (T,)
        G[-1] = r[-1]
        for i in range(T-2,-1,-1):
            G[i] = G[i+1]*gamma + r[i]
        
        # Generate one hot vector for G
        # to use category cross-entropy loss
        in_shape = env.action_space.n
        Ghot = np.zeros((T,in_shape))
        for i in range(T):
            Ghot[i,actions[i]] += G[i]/T
                
        self.model.fit(np.array(states),Ghot,verbose = 0)


    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        
        done = False
        state = env.reset()
        while not done:
            act_probs = self.model.predict(state.reshape(1,-1)).squeeze() # evaluate policy
            act = np.random.choice(np.arange(act_probs.shape[0]), p = act_probs)
            next_state, r, done, _ = env.step(act)
            states.append(state)
            actions.append(act)
            rewards.append(r)
            state = next_state.copy()
        return states, actions, rewards
    
    def test(self,env,gamma = 1):
        # Test the model 
        reward = []
        for i in range(20):
            state = env.reset()
            done = False
            r_episode = 0
            # Check if the game is terminated
            while done == False:
                # Take action and observe
                act_probs = self.model.predict(state.reshape(1,-1)).squeeze() # evaluate policy
                act = np.argmax(act_probs)
                
                next_state,r,done,_ = env.step(act)
                r_episode += r
            
            reward.append(r_episode)
            
        reward_mean = np.mean(np.array(reward))
        return reward_mean
    
    

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()



        
def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render
    
    # Create the environment.
    env = gym.make('LunarLander-v2')
    # TODO: Create the model.
    in_shape = env.observation_space.shape[0]
    
    
    model = Sequential()       # the model that is trained for Q value estimator (Q hat)
    model.add(Dense(16,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),
                    bias_initializer='zeros', 
                    activation='relu', 
                    input_shape=(in_shape,)))  # input: state and action
#    model.add(Dense(16,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),
#                    bias_initializer='zeros',
#                    activation='relu'))
    model.add(Dense(16,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),
                    bias_initializer='zeros',
                    activation='relu'))
    model.add(Dense(4,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),
                    bias_initializer='zeros', 
                    activation='softmax'))
    
    # TODO: Train the model using REINFORCE and plot the learning curve.
    reinforce_model = Reinforce(model)
    episode = 0
    r = 0
    while r < 200:
        reinforce_model.train(env)
        episode += 1
        if episode % 200 == 0:
            r = reinforce_model.test(env)
            print('------------------------')
            print('Episode: ',episode)
            print('Reward: ',r)

# train on batches, or do normalization of reward
        
        

if __name__ == '__main__':
    main(sys.argv)
