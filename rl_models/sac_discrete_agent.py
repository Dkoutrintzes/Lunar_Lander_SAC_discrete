import torch
import numpy as np
from rl_models.networks_discrete import update_params, Actor, Critic, ReplayBuffer
import torch.nn.functional as F
import os
import random
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)


class DiscreteSACAgent:
    def __init__(self, config=None, alpha=0.05, beta=0.0003, input_dims=[8],
                 env=None,lr = 0.0003, gamma=0.99, n_actions=2, buffer_max_size=50000, tau=0.005,
                 update_interval=1, layer1_size=64, layer2_size=64, batch_size=64, reward_scale=2,
                 chkpt_dir='tmp/sac',load_file=None, target_entropy_ratio=0.4,participant_name=None):

        self.env = env

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size

        self.update_interval = update_interval
        self.buffer_max_size = buffer_max_size
        self.scale = reward_scale
        self.lr = lr
        self.env = env
        self.input_dims = input_dims[0]
        self.n_actions = n_actions
        self.p_name = participant_name
        self.weight = 1

        # EXTRA 
        self.use_clip_q = True
        self.clip_q_epsilon = 0.5
        self.use_avg_q = True
        self.use_entropy_target = True
        self.entropy_penalty_beta = 0.5

        self.chkpt_dir = chkpt_dir

        if not os.path.exists(self.chkpt_dir):
            os.makedirs(self.chkpt_dir)

        self.action_space = self.n_actions
        print("The action space is ", self.action_space)
        self.target_entropy = 0.98 * np.log(np.prod(self.action_space))  # -np.prod(action_space.shape)
        self.model_name = 'sac_' + str(random.randint(0, 9)) + str(random.randint(0, 9))
        # Saving arrays

        print("The target entropy is ",self.target_entropy)

        # if config is not None and 'chkpt_dir' in config["SAC"].keys():
        #     self.chkpt_dir = config['chkpt_dir']
        print('Inp_dims: ', self.input_dims)
        self.actor = Actor(self.input_dims, self.n_actions, self.layer1_size,name=self.model_name, chkpt_dir=self.chkpt_dir).to(device)
        self.critic = Critic(self.input_dims, self.n_actions, self.layer1_size,name=self.model_name, chkpt_dir=self.chkpt_dir).to(device)
        self.target_critic = Critic(self.input_dims, self.n_actions, self.layer1_size,name=self.model_name, chkpt_dir=self.chkpt_dir).to(
            device)
        print(self.actor)
        self.target_critic.load_state_dict(self.critic.state_dict())


        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-4)
        self.critic_q1_optim = torch.optim.Adam(self.critic.qnet1.parameters(), lr=self.lr, eps=1e-4)
        self.critic_q2_optim = torch.optim.Adam(self.critic.qnet2.parameters(), lr=self.lr, eps=1e-4)

        # target -> maximum entropy (same prob for each action)
        # - log ( 1 / A) = log A
        # self.target_entropy = -np.log(1.0 / action_dim) * self.target_entropy_ratio
        # self.target_entropy = np.log(action_dim) * self.target_entropy_ratio

        #self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.log_alpha = torch.tensor(alpha, requires_grad=True, device=device)
        #self.log_alpha.exp() = 0.05

        print('Alpha: ', self.log_alpha)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4, eps=1e-4)

        self.memory = ReplayBuffer(self.buffer_max_size)
    
    def learn(self,block_number, interaction=None):
        if interaction is None:
            
            states, actions, rewards, states_, dones,transition_info = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, states_, dones,transition_info = interaction
            states, actions, rewards, states_, dones,transition_info = [np.asarray([states]), np.asarray([actions]),
                                                        np.asarray([rewards]), np.asarray([states_]),
                                                        np.asarray([dones]), np.asarray([transition_info])]

        states = torch.from_numpy(states).float().to(device)
        states_ = torch.from_numpy(states_).float().to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1)  # dim [Batch,] -> [Batch, 1]
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)

        batch_transitions = states, actions, rewards, states_, dones

        weights = 1.  # default
        q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch_transitions, weights)
        policy_loss, entropies, q1, q2, action_probs = self.calc_policy_loss(batch_transitions, weights)
        #print(q1,q2)
        entropy_loss = self.calc_entropy_loss(entropies, weights)

        self.critic_q1_optim.zero_grad()
        self.critic_q2_optim.zero_grad()
        self.actor_optim.zero_grad()
        #self.alpha_optim.zero_grad()

        q1_loss.backward()
        q2_loss.backward()
        policy_loss.backward()
        #entropy_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)

        self.critic_q1_optim.step()
        self.critic_q2_optim.step()
        self.actor_optim.step()
        #self.alpha_optim.step()

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), entropy_loss.item(), entropies.mean().item(), mean_q1, mean_q2

    def add_point(self):
        self.alpha_hisotry.append(0)
        self.alpha_hisotry.append(1)

    def get_alpha_history(self):
        return self.alpha_history

    def update_target(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def soft_update_target(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states)
        curr_q1 = curr_q1.gather(1, actions)  # select the Q corresponding to chosen A
        curr_q2 = curr_q2.gather(1, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            action_probs = self.actor(next_states)
            z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
            log_action_probs = torch.log(action_probs + z)

            next_q1, next_q2 = self.target_critic(next_states)
            # next_q = (action_probs * (
            #     torch.min(next_q1, next_q2) - self.alpha * log_action_probs
            # )).mean(dim=1).view(self.memory_batch_size, 1) # E = probs T . values

            alpha = self.log_alpha.exp()
            next_q = action_probs * (torch.min(next_q1, next_q2) - alpha * log_action_probs)
            next_q = next_q.sum(dim=1)

            target_q = rewards + (1 - dones) * self.gamma * (next_q)
            return target_q.unsqueeze(1)

 
    def calc_critic_loss(self, batch, weights):
        target_q = self.calc_target_q(*batch)
        curr_q1, curr_q2 = self.calc_current_q(*batch)
        # TD errors for updating priority weights
        # errors = torch.abs(curr_q1.detach() - target_q)
        errors = None
        mean_q1, mean_q2 = None, None

        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        # Critic loss is mean squared TD errors with priority weights.
        # q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        # q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)

        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)

        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch

        # (Log of) probabilities to calculate expectations of Q and entropies.
        action_probs = self.actor(states)
        z = (action_probs == 0.0).float() * 1e-8  # for numerical stability
        log_action_probs = torch.log(action_probs + z)

        # with torch.no_grad():
        # Q for every actions to calculate expectations of Q.
        # q1, q2 = self.critic(states)
        # q = torch.min(q1, q2)

        q1, q2 = self.critic(states)

        alpha = self.log_alpha.exp()
        # minq = torch.min(q1, q2)
        # inside_term = alpha * log_action_probs - minq
        # policy_loss = (action_probs * inside_term).mean()

        # Expectations of entropies.
        entropies = - torch.sum(action_probs * log_action_probs, dim=1)
        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy) with
        # priority weights.
        policy_loss = (weights * (- q - alpha * entropies)).mean()  # avg over Batch

        return policy_loss, entropies, q1, q2, action_probs

    def calc_entropy_loss2(self, pi_s, log_pi_s):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        alpha = self.log_alpha.exp()
        inside_term = - alpha * (log_pi_s + self.target_entropy).detach()
        entropy_loss = (pi_s * inside_term).mean()
        return entropy_loss

    def calc_entropy_loss(self, entropies, weights):
        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies).detach()
            * weights)
        return entropy_loss

    def save_models(self):
        if self.chkpt_dir is not None:
            print('.... saving models ....')
            self.actor.save_checkpoint()
            self.critic.save_checkpoint()
            self.target_critic.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
    
    