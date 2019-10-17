# run this script to recover from last crashed point
# load saved models

import numpy as np
import gym
import matplotlib.pyplot as plt

#from reinforce import Reinforce
import torch

import a2c


num_episodes_remain = 13000    # remaining episodes to run
episodes_run = 7000
lr = 6e-4
critic_lr = 1.5e-4
n = 50  

env = gym.make('LunarLander-v2')
in_shape = env.reset().reshape((-1,1)).shape[0]
out_shape =env.action_space.n

policy_reload = a2c.Model_actor(in_shape,out_shape)
critic_reload = a2c.Model_critic(in_shape,1)

policy_reload.load_state_dict(torch.load('../a2c_actor_model'))
critic_reload.load_state_dict(torch.load('../a2c_critic_model'))

a2c_class = a2c.A2C(policy_reload, lr, critic_reload, critic_lr,n)

# temp: based on command window logging
tempr = [-155,-134,-111,-158,-131,-133,-134,-139,-126,-101,-92,-82,-118,-97,-87,-74,-78,-82,-54,-78,-54,-67,-57,-26,-38,-36,-42,0.134,-8,-22,9.4]
tempx = (np.arange(31)+1)*200
temp_error = [136,74,71,91,42,48,75,58,55,53,40,40,56,36,51,34,50,61,51,68,25,45,65,30,50,58,57,39,47,73,53]
assert len(temp_error) == len(tempr)

iteration = episodes_run
error_recover = temp_error.copy() # np.load(PATH)
x_recover = tempx.tolist()
y_recover = tempr.copy()  # np.load(PATH)

batch_size = 10; # how much more data episodes per training call
x0 = episodes_run/batch_size


while iteration - episodes_run < num_episodes_remain:
    a2c_class.train(env,batch_size)
    if iteration % (200/batch_size) == 0:
        r,r_list = a2c_class.test(env)
        print('Reward: ',r)
        stdev = np.std(r-r_list)
        print('Deviation: ',stdev)
        print('Iteration: ',iteration*batch_size)
        error_recover.append(stdev)
        x0 += 1
        y_recover.append(r)
        x_recover.append(x0*batch_size)
        torch.save(policy_reload.state_dict(), '../a2c_actor_model')
        torch.save(critic_reload.state_dict(), '../a2c_critic_model')  
    np.save('data_xy_e_a2c_N='+str(n),[x_recover,y_recover,error_recover]) 
    iteration += 1
plt.errorbar(x_recover,y_recover,error_recover,ecolor='r')     
plt.savefig('reward_plot_a2c_N='+str(n)+'.png')  


plt.show()
    
