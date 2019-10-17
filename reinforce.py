
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
from keras import initializers
from keras import losses
from keras import Sequential
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

#import torch
#import torch.nn as nn
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr=1e-4):
        self.model = model
        self.model.compile(loss=losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=lr))
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.


    def train(self, env, gamma=0.99):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        
        states, actions, rewards = self.generate_episode(env)
#        for i in range(5):
#            ss,aa,rr = self.generate_episode(env)
#            states.extend(ss)
#            actions.extend(aa)
#            rewards.extend(rr)

        n = len(states)
        G = []
        
        r = np.array(rewards)
        G = np.zeros_like(r)
        G[-1] = r[-1]                  # calculate G by summing up r
        for i in range(n-2,-1,-1):
            G[i] = G[i+1]*gamma + r[i]
        
        G = (G.copy() - np.mean(G))/np.var(G)**0.5


       # for i in range(n-1,-1,-1):  # counting backwards index  
       #     G.append(np.sum([rewards[j]*np.power(gamma,j-i) for j in range(i,1,n)])) 
        
        # one hot vector for G
        in_shape = env.action_space.n
        Ghot = np.zeros((n,in_shape))
 #      print('hot vector shape: ',Ghot.shape)
        for i in range(n):
            Ghot[i,actions[i]] = G[i]
        assert Ghot.shape[1] == in_shape
                    
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
            states.append(state)
            actions.append(act)
            next_state,r,done,_ = env.step(act)
            state = next_state.copy()
            rewards.append(r)
        env.reset()
        return states, actions, rewards
    
    def test(self,env,gamma = 0.99):
        # Test the model 
        reward = []
        run_count = []
        for i in range(20):
            state = env.reset()
            done = False
            r_episode = 0
            run_c = 0
            # Check if the game is terminated
            while done == False:
                # Take action and observe
                act_probs = self.model.predict(state.reshape(1,-1)).squeeze() # evaluate policy
                act = np.random.choice(np.arange(act_probs.shape[0]), p = act_probs)
                #act = np.argmax(act_probs)
               
                next_state,r,done,_ = env.step(act)
                state = next_state.copy()
                r_episode += r
                run_c += 1
            run_count.append(run_c)
            reward.append(r_episode)

        reward_mean = np.mean(np.array(reward))
        print('------------------------')
        print('average test length',np.mean(np.array(run_count)))
        print('Max reward',np.max(np.array(reward)))
        return reward_mean, np.array(reward)
    
    

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



        
#def main(args):
    # Parse command-line arguments.
#    args = parse_arguments()
#    num_episodes = args.num_episodes
#    lr = args.lr
#    render = args.render
    
    # Create the environment.
    # LunarLander-v2
env = gym.make('LunarLander-v2')    
#env = gym.make('CartPole-v0')    

# TODO: Create the model.
in_shape = env.observation_space.shape[0]
out_shape = env.action_space.n

model = Sequential()       # the model that is trained for Q value estimator (Q hat)
model.add(Dense(30,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros', activation='tanh', input_shape=(in_shape,)))  # input: state and action
#model.add(BatchNormalization())
model.add(Dense(25,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros',activation='tanh'))
#model.add(BatchNormalization())
model.add(Dense(16,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros',activation='tanh'))
#model.add(BatchNormalization())
model.add(Dense(out_shape,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros', activation='softmax'))


# TODO: Train the model using REINFORCE and plot the learning curve.
reinforce_model = Reinforce(model,lr=2e-4)
r = 0
iteration = 0
episode_num = 15000
error = []
y = []
x = []
x0 = 1
while iteration < episode_num:
    reinforce_model.train(env)
    if iteration % 100 == 0:
        r,r_list = reinforce_model.test(env)
        print('Reward: ',r)
        print('Iteration: ',iteration)
        error.append(np.max(r-r_list))
        y.append(r)
        x.append(x0)
        x0 +=1
    iteration += 1
plt.errorbar(x,y,error,ecolor='r')
plt.savefig('reward_plot_reinforce.png')      
plt.show()

#if __name__ == '__main__':
#    main(sys.argv)
