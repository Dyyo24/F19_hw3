import a2c.py
import numpy as np
import gym

#matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from reinforce import Reinforce
import torch



num_episodes = 15000

lr = 6e-4
critic_lr = 1.5e-4
n = 50

# Create the environment.
env = gym.make('LunarLander-v2')
#env = gym.make('CartPole-v0') 
in_shape = env.reset().reshape((-1,1)).shape[0]
out_shape =env.action_space.n

policy = a2c.Model_actor(in_shape,out_shape)
critic_model = a2c.Model_critic(in_shape,1)

a2c_class = a2c.A2C(policy, lr, critic_model, critic_lr,n)

r = 0
iteration = 0
error = []
y = []
x = []
x0 = 1
batch_size = 10; # how much more data episodes per training call
episode_num = num_episodes/batch_size

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
        x.append(x0*batch_size)
        torch.save(policy.state_dict(), '../a2c_actor_model')
        torch.save(critic_model.state_dict(), '../a2c_critic_model')        
    iteration += 1
    np.save('data_xy_e_a2c_N='+str(n),[x,y,error]) 
plt.errorbar(x,y,error,ecolor='r')     
plt.savefig('reward_plot_a2c_N='+str(n)+'.png')  
plt.show()


#Load model
#model = TheModelClass(in_shape,out_shape)
#model.load_state_dict(torch.load(PATH))
#model.eval()

