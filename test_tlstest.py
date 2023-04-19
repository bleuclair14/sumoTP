# Team Project: Toward Sustainable Cities through simulation
# Reinforcement Learning
# testing the trained agent

# import Environment:
from env_tlstest import TLSEnv

# import algorithms from stable baselines:
from stable_baselines3 import PPO, A2C, DQN # noqa

# to create paths:
from os.path import join
import sys
"""
args:
1: saved model name

"""
num_args = len(sys.argv)
if num_args>=2:
    final_model=sys.argv[1]
else: 
    raise ValueError("please indicate the saved model name")
env = TLSEnv("2j",cmd=True)  # create Environment

# set paths and load model:
save_path = join("Training", "Models", final_model)
log_path = join("Training", "Logs")
model = PPO.load(save_path, env)

# Test the trained agent:
obs = env.reset()
n_steps = 2000
done = False
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=False)
    print(f"Step {step+1}")
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    if done:
        break

env.close()
