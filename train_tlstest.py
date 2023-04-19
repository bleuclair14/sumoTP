# Team Project: Toward Sustainable Cities through simulation
# training the env RL

from env_tlstest import TLSEnv

from stable_baselines3 import PPO, A2C, DQN

from os.path import join
import sys
"""
args:
1: net file name

"""
num_args = len(sys.argv)
if num_args>=2:
    NET = sys.argv[1]
else:
    NET = "small"
NR_STEPS = 100000

env = TLSEnv(NET, cmd=True)

save_path = join("Training", "Models", f"PPO_{NR_STEPS}_2j")
log_path = join("Training", "Logs")


model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(NR_STEPS)
model.save(save_path)
