#import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from reinforce import Reinforce
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch
torch.manual_seed(1)

class A2C():
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
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
        self.lr = lr
        self.critic_lr = critic_lr
        self.scale_factor = 1e-2
        
    def policy_loss(self,states,actions,Rt):   # input numpy array of Rt and list of states
        # maximize actor probabililty of advantage
        # track the gradient related to states
        states = torch.autograd.Variable(torch.Tensor(states), requires_grad = True)
        act_prob = self.model.forward(states)

        Vw = self.critic_model.forward(torch.Tensor(states)).double().squeeze().detach()   # V(st)

        R = torch.from_numpy(Rt).double().detach()      
        in_shape = list(act_prob.size())[1]
        
        num = Rt.shape[0]
        advantage_hot_vect = torch.zeros([num,in_shape]).double()
        
        assert len(actions) == num
        for i in range(num):
            advantage_hot_vect[i,actions[i]] = (R-Vw)[i]

        assert list(act_prob.size())[0] == num
        loss_list = advantage_hot_vect@torch.t(torch.log(act_prob)).double()
        assert list(loss_list.size())[0] == num        
        loss = torch.mean( -1*loss_list) # minus for maximum of the expression

        return loss

    def critic_loss(self,states,Rt):   # input numpy array of Rt and list of states
        # minimize critic prediction error
        states = torch.autograd.Variable(torch.Tensor(states), requires_grad = True)
        R = torch.from_numpy(Rt).double().detach()
        Vw = self.critic_model.forward(states).double()

        num = Rt.shape[0]
        assert list((R-Vw).size())[0] == num    
        loss = torch.mean( (R-Vw)*(R-Vw) )

        return loss

    def train(self, env, batch_size, gamma=0.99):
        # Trains the model on a single episode using A2C.
        
        states=[]
        actions=[]
        rewards = []
        for i in range(batch_size):
            ss,aa,rr = self.generate_episode(env)
            states.extend(ss)
            actions.extend(aa)
            rewards.extend(rr)
            
        T = len(states)
        R = np.zeros((T,))
        R[-1] = rewards[-1]
        gamma_matrix = gamma**np.arange(self.n)
        
        for i in range(T-2,-1,-1):
            if i + self.n < T:
                Vend = self.critic_model.forward(torch.from_numpy(states[i+self.n]))
                Vend = Vend.item()
                r_matrix = rewards[i:i+self.n].copy()
                assert len(r_matrix) == self.n
                R[i] = (gamma**self.n)*Vend + np.sum(r_matrix*gamma_matrix)
                
            elif i + self.n >= T:
                R[i] = R[i+1]*gamma + rewards[i]
        
        R *= self.scale_factor # downscale rewards
        
        # gradient descent on policy actor 
        optimizer1 = optim.Adam(self.model.parameters(),lr = self.lr)
        optimizer1.zero_grad() 
        Loss_policy = self.policy_loss(states,actions,R) 
        Loss_policy.backward()
        optimizer1.step()

        
        # gradient descent on critic
        optimizer2 = optim.Adam(self.critic_model.parameters(),lr = self.critic_lr)
        optimizer2.zero_grad() 
        Loss_critic = self.critic_loss(states,R)
        Loss_critic.backward()        
        optimizer2.step()
      
        
        
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
            act_probs = self.model.forward(torch.from_numpy(state)) # evaluate policy
            act_probs = act_probs.detach().numpy().squeeze()
            act = np.random.choice(np.arange(env.action_space.n), p = act_probs)
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
                act_probs = self.model.forward(torch.from_numpy(state)) # evaluate policy

                act_probs = act_probs.detach().numpy()
                act = np.random.choice(np.arange(act_probs.shape[0]), p = act_probs)
                #act = np.argmax(act_probs.detach().numpy())                
                next_state,r,done,_ = env.step(act)
                state = next_state.copy()
                r_episode += r
                run_c += 1
            run_count.append(run_c)
            reward.append(r_episode)
            env.reset()
        reward_mean = np.mean(np.array(reward))
        print('------------------------')
        print('average test length',np.mean(np.array(run_count)))
        print('Max reward',np.max(np.array(reward)))
        return reward_mean,np.array(reward)    



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



class Model_actor(nn.Module):

    def __init__(self,in_shape,out_shape):
        
        super(Model_actor, self).__init__()
        #self.h1 = nn.utils.weight_norm(nn.Linear(in_shape,16),name='weight')
        self.h1 = nn.Linear(in_shape,40)
        torch.nn.init.xavier_uniform_(self.h1.weight, gain=1/np.sqrt(2))
        self.h1.bias.data.fill_(0)
        
        #self.h2 = nn.utils.weight_norm(nn.Linear(16,16),name='weight')
        self.h2 = nn.Linear(40,32)
        torch.nn.init.xavier_uniform_(self.h2.weight, gain=1/np.sqrt(2))
        self.h2.bias.data.fill_(0)
        
        #self.h3 = nn.utils.weight_norm(nn.Linear(16,16),name='weight')
        self.h3 = nn.Linear(32,16)
        torch.nn.init.xavier_uniform_(self.h3.weight, gain=1/np.sqrt(2))
        self.h3.bias.data.fill_(0)
        
        #self.h4 = nn.utils.weight_norm(nn.Linear(16,out_shape),name='weight')
        self.h4 = nn.Linear(16,out_shape)
        torch.nn.init.xavier_uniform_(self.h4.weight, gain=1/np.sqrt(2))
        self.h4.bias.data.fill_(0)
    
    def forward(self,x):
        x = F.tanh(self.h1(x.float()))

        x = F.tanh(self.h2(x))
        
        x = F.tanh(self.h3(x))

        y_pred = F.softmax(self.h4(x),dim=-1)

        return y_pred

class Model_critic(nn.Module):

    def __init__(self,in_shape,out_shape):
        
        super(Model_critic, self).__init__()
        #self.h1 = nn.utils.weight_norm(nn.Linear(in_shape,16),name='weight')
        self.h1 = nn.Linear(in_shape,40)
        torch.nn.init.xavier_uniform_(self.h1.weight, gain=1/np.sqrt(2))
        self.h1.bias.data.fill_(0)
        
        #self.h2 = nn.utils.weight_norm(nn.Linear(16,16),name='weight')
        self.h2 = nn.Linear(40,32)
        torch.nn.init.xavier_uniform_(self.h2.weight, gain=1/np.sqrt(2))
        self.h2.bias.data.fill_(0)
        
        #self.h3 = nn.utils.weight_norm(nn.Linear(16,16),name='weight')
        self.h3 = nn.Linear(32,16)
        torch.nn.init.xavier_uniform_(self.h3.weight, gain=1/np.sqrt(2))
        self.h3.bias.data.fill_(0)
        
        #self.h4 = nn.utils.weight_norm(nn.Linear(16,out_shape),name='weight')
        self.h4 = nn.Linear(16,out_shape)
        torch.nn.init.xavier_uniform_(self.h4.weight, gain=1/np.sqrt(2))
        self.h4.bias.data.fill_(0)
    
    def forward(self,x):
        x = F.tanh(self.h1(x.float()))

        x = F.tanh(self.h2(x))
        
        x = F.tanh(self.h3(x))
        
        y_pred = self.h4(x)

        return y_pred



#def main(args):
    # Parse command-line arguments.
args = parse_arguments()
num_episodes = args.num_episodes
lr = 8e-4
critic_lr = 2e-4
n = args.n
render = args.render

# Create the environment.
env = gym.make('LunarLander-v2')
#env = gym.make('CartPole-v0') 
# TODO: Create the model.
in_shape = env.reset().reshape((-1,1)).shape[0]
out_shape =env.action_space.n

policy = Model_actor(in_shape,out_shape)
critic_model = Model_critic(in_shape,1)

a2c_class = A2C(policy, lr, critic_model, critic_lr,n=20)

r = 0
iteration = 0
error = []
y = []
x = []
x0 = 1
batch_size = 10; # how much more data episodes per training call
episode_num = 30000/batch_size

while iteration < episode_num:
    a2c_class.train(env,batch_size)
    if iteration % (200/batch_size) == 0:
        r,r_list = a2c_class.test(env)
        print('Reward: ',r)
        stdev = np.std(r-r_list)
        print('Deviation: ',stdev)
        print('Iteration: ',iteration*batch_size)
        error.append(stdev)
        x0 += 1
        y.append(r)
        x.append(x0)
        torch.save(policy.state_dict(), '../actor_model')
        torch.save(critic_model.state_dict(), '../critic_model')        
    iteration += 1
plt.errorbar(x,y,error)     
plt.savefig('reward_plot_a2c.png')  
np.save('x',x) 
np.save('y',y)
np.save('error',error)
plt.show()

#Load model
#model = TheModelClass(in_shape,out_shape)
#model.load_state_dict(torch.load(PATH))
#model.eval()


    #if __name__ == '__main__':
#    main(sys.argv)
