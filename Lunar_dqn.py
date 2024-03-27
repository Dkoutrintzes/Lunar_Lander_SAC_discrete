import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import base64, io
from tqdm import tqdm
from rl_models_rev.sac_discrete_agent import DiscreteSACAgent
from rl_models.sac_discrete_agent import DiscreteSACAgent as baseDiscreteSACAgent
import numpy as np
from collections import deque, namedtuple
import os
# For visualization
from gymnasium.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import glob
import csv
import argparse
from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBufferManager,
    SequenceSummaryStats,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tqdm import tqdm
from tianshou.data import Batch
from tianshou.data import ReplayBuffer as RB
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule
from rev_discrete_sac.discrete_sac import DiscreteSACDevPolicy
from torch.distributions import Categorical
import torch.nn.functional as F
from numpy import mean
from tianshou.policy import DiscreteSACPolicy
from gymnasium.spaces import Discrete
from datetime import datetime
BUFFER_SIZE = 20000 # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 3        # how often to update the network
EVAL_EVERY = 100

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="LunarLander-v2")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--actor-lr", type=float, default=LR)
    parser.add_argument("--critic-lr", type=float, default=LR)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auto-alpha", action="store_true", default=False)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--step-per-epoch", type=int, default=BUFFER_SIZE)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=[128, 128])
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--frames-stack", type=int, default=4)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument("--avg-q", action="store_true", default=True)
    parser.add_argument('--clip-q', action="store_true", default=True)
    parser.add_argument("--clip-q-epsilon", type=float, default=0.5)
    parser.add_argument("--entropy-penalty", action="store_true", default=True)

    parser.add_argument('--entropy-penalty-beta',type=float,default=0.5)
    parser.add_argument('--savedir',type=str,default='datasave')
    parser.add_argument('--name',type=str,default='tianbasesac')

    return parser.parse_args()

env = gym.make('LunarLander-v2')
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Alpha Temparature updates is set to: ', get_args().auto_alpha)

class TianshouDiscreteRevSac:
    def __init__(self,args = None,env = env):
        self.args = args
        self.weight = 1 
        self.buffer_current_size = 0
        self.args.state_shape = env.observation_space.shape or env.observation_space.n 
        self.args.action_shape = env.action_space.shape or env.action_space.n
        print(self.args.state_shape,self.args.action_shape)
        # model
        Q_param = {"hidden_sizes": self.args.hidden_size}
        V_param = {"hidden_sizes": self.args.hidden_size}
        
        net = Net(
            self.args.state_shape,
            self.args.action_shape,
            hidden_sizes=self.args.hidden_sizes,
            device=self.args.device,
            dueling_param=(Q_param, V_param),
        ).to(self.args.device)

        actor = Actor(net, self.args.action_shape, device=self.args.device, softmax_output=False)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=self.args.actor_lr)
        critic1 = Critic(net, last_size=self.args.action_shape, device=self.args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=self.args.critic_lr)
        critic2 = Critic(net, last_size=self.args.action_shape, device=self.args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=self.args.critic_lr)

        # define policy
        if self.args.auto_alpha:
            target_entropy = 0.98 * np.log2(np.prod(self.args.action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=self.args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=self.args.alpha_lr)
            self.args.alpha = (target_entropy, log_alpha, alpha_optim)
        
        self.policy = DiscreteSACDevPolicy(
            actor=actor,
            actor_optim=actor_optim,
            critic1=critic1,
            critic1_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            action_space=Discrete(self.args.action_shape),
            tau=self.args.tau,
            gamma=self.args.gamma,
            alpha=self.args.alpha,
            estimation_step=self.args.n_step,

            use_avg_q=args.avg_q,
            use_clip_q=args.clip_q,
            clip_q_epsilon=args.clip_q_epsilon,
            use_entropy_penalty=args.entropy_penalty,
            entropy_penalty_beta=args.entropy_penalty_beta
        ).to(self.args.device)
        
        # replay buffer
        self.replay_buffer = RB(size=self.args.buffer_size)

        

    def sample_action(self,obs):
        with torch.no_grad():
            result = self.policy(Batch(obs=[obs],info={}))
            #print(result)
            actpolicy = result.get("policy",Batch())
            assert isinstance(actpolicy, Batch)

            lstate = result.get("state",None)
            if lstate is not None:
                self.policy.hidden_state = lstate
    
            act = to_numpy(result.act[0])
            old_entropy = result.get('dist').entropy()
            old_entropy = to_numpy(old_entropy)
            action_remap = self.policy.map_action(act)

            return action_remap, old_entropy, actpolicy

    def seve_to_buffer(self,actpolicy,obs,action,reward,next_obs,terminated,truncated,done,old_entropy):
        self.replay_buffer.add(Batch(policy = actpolicy,obs=obs, act=action, rew=reward, done=done,terminated=terminated,truncated=truncated, obs_next=next_obs, old_entropy=old_entropy, info={}))
        self.buffer_current_size += 1
        return

    def learn(self):
        result = self.policy.update(self.args.batch_size, self.replay_buffer)
        train_time = result.train_time
        actor_loss = result.actor_loss
        critic1_loss = result.critic1_loss
        critic2_loss = result.critic2_loss
        alpha_loss = result.alpha_loss
        alpha = result.alpha
        
        return train_time, actor_loss, critic1_loss, critic2_loss, alpha_loss, alpha

    def save_model(self,name):
        torch.save(self.policy.state_dict(), self.chkpt_dir +name+ '.pt')
    
    def load_model(self):
        self.policy.load_state_dict(torch.load(self.load_file))

class TianshouDiscreteSac:
    def __init__(self,args = None,env = env):
        self.args = args
        self.weight = 1 
        self.buffer_current_size = 0
        self.args.state_shape = env.observation_space.shape or env.observation_space.n 
        self.args.action_shape = env.action_space.shape or env.action_space.n
        print(self.args.state_shape,self.args.action_shape)
        # model
        Q_param = {"hidden_sizes": self.args.hidden_size}
        V_param = {"hidden_sizes": self.args.hidden_size}
        
        net = Net(
            self.args.state_shape,
            self.args.action_shape,
            hidden_sizes=self.args.hidden_sizes,
            device=self.args.device,
            dueling_param=(Q_param, V_param),
        ).to(self.args.device)

        actor = Actor(net, self.args.action_shape, device=self.args.device, softmax_output=False)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=self.args.actor_lr)
        critic1 = Critic(net, last_size=self.args.action_shape, device=self.args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=self.args.critic_lr)
        critic2 = Critic(net, last_size=self.args.action_shape, device=self.args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=self.args.critic_lr)

        # define policy
        if self.args.auto_alpha:
            target_entropy = 0.98 * np.log2(np.prod(self.args.action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=self.args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=self.args.alpha_lr)
            self.args.alpha = (target_entropy, log_alpha, alpha_optim)
        
        self.policy = DiscreteSACPolicy(
            actor=actor,
            actor_optim=actor_optim,
            critic=critic1,
            critic_optim=critic1_optim,
            critic2=critic2,
            critic2_optim=critic2_optim,
            action_space=Discrete(self.args.action_shape),
            tau=self.args.tau,
            gamma=self.args.gamma,
            alpha=self.args.alpha,
            estimation_step=self.args.n_step,
        ).to(self.args.device)
        
        # replay buffer
        self.replay_buffer = RB(size=self.args.buffer_size)

    def sample_action(self,obs):
        with torch.no_grad():
            result = self.policy(Batch(obs=[obs],info={}))
            #print(result)
            actpolicy = result.get("policy",Batch())
            assert isinstance(actpolicy, Batch)

            lstate = result.get("state",None)
            if lstate is not None:
                self.policy.hidden_state = lstate
    
            act = to_numpy(result.act[0])
            old_entropy = result.get('dist').entropy()
            old_entropy = to_numpy(old_entropy)
            action_remap = self.policy.map_action(act)

            return action_remap, old_entropy, actpolicy

    def seve_to_buffer(self,actpolicy,obs,action,reward,next_obs,terminated,truncated,done,old_entropy):
        self.replay_buffer.add(Batch(policy = actpolicy,obs=obs, act=action, rew=reward, done=done,terminated=terminated,truncated=truncated, obs_next=next_obs, old_entropy=old_entropy, info={}))
    
    def learn(self):
        result = self.policy.update(self.args.batch_size, self.replay_buffer)
        train_time = result.train_time
        actor_loss = result.actor_loss
        critic1_loss = result.critic1_loss
        critic2_loss = result.critic2_loss
        alpha_loss = result.alpha_loss
        alpha = result.alpha
        
        return train_time, actor_loss, critic1_loss, critic2_loss, alpha_loss, alpha

class LunarLanderTest:
    def __init__(self, env, policy, agent):
        self.env = gym.make('LunarLander-v2')
        self.policy = policy
        self.agent = agent

        self.train_scores = []
        self.eval_scores = []

        self.train_data = []
        self.scores_window = deque(maxlen=100)

    def run_lunar_games(self, max_iterations=30000, max_t=1000):
        game = 0
        total_iteraction = 0

        while total_iteraction < max_iterations:
            score, iteractions = self.play(max_t, eval_game=False)
            total_iteraction += iteractions

            self.train_scores.append([int(score)])
            self.scores_window.append(score)
            game += 1

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(game, np.mean(self.scores_window)), end="")
            if game % EVAL_EVERY == 0:
                score_eval, iteractions = self.play(max_t, eval_game=True)
                self.eval_scores.append([int(score_eval)])
                print('Game: ', game, 'Score: ', score)

    def play(self, max_t, eval_game=False):
        state = env.reset()
        if type(state) is tuple:
            state = state[0]
        score = 0
        step = 0
        iteractions = 0
        for t in range(max_t):
            if self.agent == 'revsac' or self.agent == 'tianbasesac':
                action, old_entropy, actpolicy = self.get_action(eval_game, state)
            else:
                action = self.get_action(eval_game,state)

            
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                done = True
            else:
                done = False
        
            if self.agent == 'revsac' or self.agent == 'tianbasesac':
                data = [state, action, reward, next_state, done, terminated, truncated, old_entropy, actpolicy]
            else:
                data = [state, action, reward, next_state, done]

            if not eval_game:
                self.save_iteraction(data)

                step = (step + 1) % UPDATE_EVERY
                if step == 0:
                    self.learn(t)
            
            state = next_state
            score += reward
            iteractions += 1
            if done:
                break

        return score, iteractions

    def save_iteraction(self, data):
        if self.agent == 'basesac':
            state, action, reward, next_state, done = data
            self.policy.memory.add(state, action, reward, next_state, done, 'transition_info')
        if self.agent == 'revsac' or self.agent == 'tianbasesac':
            state, action, reward, next_state, done, terminated, truncated, old_entropy, actpolicy = data
            self.policy.seve_to_buffer(actpolicy, state, action, reward, next_state, terminated, truncated, done, old_entropy)

    def get_action(self, eval_game=False, state=None):
        with torch.no_grad():
            if self.agent is None:
                action = random.choice(np.arange(env.action_space.n))
            elif self.agent == 'basesac':
                
                action,argmax = self.policy.actor.sample_act(state)
                if eval_game:
                    return argmax

            if self.agent == 'revsac' or self.agent == 'tianbasesac':
                if eval_game:
                    self.policy.policy.eval()
                action_remap, old_entropy, actpolicy = self.policy.sample_action(state)
                if eval_game:
                    self.policy.policy.train()
                return action_remap, old_entropy, actpolicy        

        return action

    def learn(self, t):
        if self.agent == 'basesac':
            result = self.policy.learn(t)
            self.train_data.append(result)
            self.policy.update_target()
        if self.agent == 'revsac' or self.agent == 'tianbasesac':
            result = self.policy.learn()
            self.train_data.append(result)

def save_csv(scores, name,path):
    folder = path
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(os.path.join(folder, name), 'w') as f:
        writer = csv.writer(f)
        for row in scores:
            writer.writerow(row)

def get_name(path,name,typefile):
    filename = name+'.'+typefile
    c=0
    while os.path.exists(os.path.join(path,filename)):
        c+=1
        filename = name+str(c)+'.'+typefile
    return filename

def save_score(scores, name):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    # save plt to image
    folder = 'images'
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.savefig(os.path.join(folder, name))




if __name__ == '__main__':
    args = get_args()
    if args.name == 'basesac':
        policy = baseDiscreteSACAgent(env=env,input_dims=env.observation_space.shape, n_actions=env.action_space.n, gamma=GAMMA, lr=LR, tau=TAU,
                            batch_size=BATCH_SIZE,alpha=args.alpha)
        game = LunarLanderTest(env,policy,'basesac')
        game.run_lunar_games(max_iterations=args.total_steps)
    if args.name == 'revsac':
        policy = TianshouDiscreteRevSac(args=args)
        game = LunarLanderTest(env,policy,'revsac')
        game.run_lunar_games(max_iterations=args.total_steps)
    if args.name == 'tianbasesac':
        policy = TianshouDiscreteSac(args=args)
        game = LunarLanderTest(env,policy,'tianbasesac')
        game.run_lunar_games(max_iterations=args.total_steps)
    
    
    folder = 'results'
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_folder = os.path.join(folder, args.name+'_'+datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_csv(game.train_scores,'train_scores.csv',save_folder)
    save_csv(game.eval_scores,'eval_scores.csv',save_folder)
    save_csv(game.train_data,'train_data.csv',save_folder)
    

    











