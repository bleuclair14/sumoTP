# Team Project: Toward Sustainable Cities through simulation
# continue training

from env_tlstest import TLSEnv
from stable_baselines3 import PPO, A2C, DQN
from os.path import join
import sys
num_args = len(sys.argv)
"""
args:
1: net file name
2: saved model name
3: number of continue-training steps

"""
if num_args>=2:
    NET = sys.argv[1]
    if num_args>=3:
        NAME_MODEL =sys.argv[2]
else:
    NET = "small"
NR_STEPS = sys.argv[3]  # number of training steps
# NAME_MODEL = "PPO_50000_continued"  # name of model to continue training on

env = TLSEnv(NET, cmd=False)  # set environment

# Paths:
save_path = join("Training", "Models", NAME_MODEL)
save_path_new = join("Training", "Models", f"{NAME_MODEL}_continued")
log_path = join("Training", "Logs")

# load, train, save:
model = PPO.load(save_path, env, verbose=1, tensorboard_log=log_path)
model.learn(NR_STEPS)
model.save(save_path_new)
