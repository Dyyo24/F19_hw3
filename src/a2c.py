import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, critic_model, lr=0.001, critic_lr=0.001, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.n = n

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, rewards = self.generate_episode(env)
        print(states.shape)
        r = np.array(rewards)
        V = np.zeros_like(r)
        T = len(states)
        for i in range(T-2,-1,-1):
            if i+self.n >= T:
                V[-1] = 0
            else:
                V[i+self.n] = self.critic_model.predict(states.reshape(1,-1)).squeeze()
                
        return


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

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


# =============================================================================
# def main(args):
#     # Parse command-line arguments.
#     args = parse_arguments()
#     num_episodes = args.num_episodes
#     lr = args.lr
#     critic_lr = args.critic_lr
#     n = args.n
#     render = args.render
# =============================================================================

    # Create the environment.
env = gym.make('LunarLander-v2')

    # TODO: Create the model.
in_shape = env.observation_space.shape[0]
actor = Sequential()       # REINFORCE model as the actor
actor.add(Dense(16,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros', activation='relu', input_shape=(in_shape,)))  # input: state and action
actor.add(Dense(16,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros',activation='relu'))
actor.add(Dense(16,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros',activation='relu'))
actor.add(Dense(4,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros', activation='softmax'))

critic = Sequential()       # The critic, output should be the value of the state
critic.add(Dense(16,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros', activation='relu', input_shape=(in_shape,)))  # input: state and action
critic.add(Dense(16,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros',activation='relu'))
critic.add(Dense(16,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros',activation='relu'))
critic.add(Dense(1,kernel_initializer=initializers.VarianceScaling(scale=1.0,mode='fan_avg',distribution='uniform'),bias_initializer='zeros', activation='softmax'))
    # TODO: Train the model using A2C and plot the learning curves.
a2c = A2C(actor,critic)
a2c.train(env)

# =============================================================================
# if __name__ == '__main__':
#     main(sys.argv)
# =============================================================================
