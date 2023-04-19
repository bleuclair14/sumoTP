# Team Project: Toward Sustainable Cities through simulation
# testing RL

from env_tlstest import TLSEnv

from stable_baselines3 import PPO, A2C

env = TLSEnv(netfile= "data/cross.sumocfg", reward_fn="dwt")

# obs = env.reset()
# env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

# GO_LEFT = 0
# # Hardcoded best agent: always go left!
# n_steps = 20
# for step in range(n_steps):
#   print("Step {}".format(step + 1))
#   obs, reward, done, info = env.step(GO_LEFT)
#   print('obs=', obs, 'reward=', reward, 'done=', done)
#   env.render()
#   if done:
#     print("Goal reached!", "reward=", reward)
#     break

model = A2C('MlpPolicy', env, verbose=1, gamma=0.99999)
model.learn(12000)
# Test the trained agent
obs = env.reset()
n_steps = 1000
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step+1}")
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    # env.render(mode='console')
    if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
        print("Goal reached!", "reward=", reward)

env.close()