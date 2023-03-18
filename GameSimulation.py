from TetrisBattle.envs.tetris_env import TetrisDoubleEnv
from CustomAgent import Agent

env = TetrisDoubleEnv()

done = False
state = env.reset()
agent_list = [Agent(), Agent()]

while not done:
    env.render()
    action = agent_list[env.game_interface.getCurrentPlayerID()].choose_action(state)
    state, reward, done, _ = env.step(action)
    env.take_turns()