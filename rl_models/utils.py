from rl_models.sac_agent import Agent
from rl_models_rev.sac_discrete_agent import DiscreteSACAgent
from rl_models_ppo.ppo import PPO_discrete

def get_sac_agent(config, env, chkpt_dir=None,p_name="None"):
    discrete = config['SAC']['discrete']
    if discrete:
        mode = config['Experiment']['mode']
        
        buffer_max_size = config['Experiment'][mode]['buffer_memory_size']
        update_interval = config['Experiment'][mode]['games_per_block']
        scale = config['Experiment'][mode]['reward_scale']
        
        if config['game']['agent_only']:
            # up: 1, down:2, left:3, right:4, upleft:5, upright:6, downleft: 7, downright:8
            action_dim = pow(2, env.action_space.actions_number)
        else:
            action_dim = env.action_space.actions_number
        print("action_dim", action_dim)
        sac = DiscreteSACAgent(config=config, env=env, input_dims=env.observation_shape,
                               n_actions=action_dim,
                               chkpt_dir=chkpt_dir, buffer_max_size=buffer_max_size, update_interval=update_interval,
                               reward_scale=scale,participant_name=p_name)
    else:
        sac = Agent(config=config, env=env, input_dims=env.observation_shape, n_actions=env.action_space.shape,
                    chkpt_dir=chkpt_dir)
    return sac

def get_ppo_agent(config,env,chkpt_dir=None,p_name="None"):
    discrete = config['SAC']['discrete']
    if discrete:
        mode = config['Experiment']['mode']
        buffer_max_size = config['Experiment'][mode]['buffer_memory_size']
        update_interval = config['Experiment'][mode]['games_per_block']
        scale = config['Experiment'][mode]['reward_scale']
        
        if config['game']['agent_only']:
            # up: 1, down:2, left:3, right:4, upleft:5, upright:6, downleft: 7, downright:8
            action_dim = pow(2, env.action_space.actions_number)
        else:
            action_dim = env.action_space.actions_number
        print("action_dim", action_dim)
        ppo = PPO_discrete(config=config, env=env, input_dims=env.observation_shape,
                           n_actions=action_dim,
                           chkpt_dir=chkpt_dir, buffer_max_size=buffer_max_size, update_interval=update_interval,
                           reward_scale=scale,participant_name=p_name)
    else:
        ppo = Agent(config=config, env=env, input_dims=env.observation_shape, n_actions=env.action_space.shape,
                    chkpt_dir=chkpt_dir)
    return ppo