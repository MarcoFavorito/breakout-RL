import click
from breakout_env.wrappers.observation_wrappers import *
from breakout_env.wrappers.reward_wrappers import *

from Breakout import BreakoutS
from BreakoutRA import BreakoutSRA, BreakoutNRA, BreakoutSRAExt
from breakout_RL.agents.NNAgent import NNAgent
from breakout_RL.agents.RLAgent import RLAgent

from breakout_RL.brains.TDBrain import QLearning, Sarsa
from breakout_RL.exploration_policies.RandomPolicyWithDecay import RandomPolicyWithDecay
from breakout_RL.misc.Renderer import Renderer
from breakout_RL.misc.StatsManager import StatsManager
from breakout_RL.utils import ID2ACTION

num_episodes = 20001

conf = {
    "observation": "number_discretized",
    # "observation": "number",
    "paddle_speed": 3,
    'paddle_width': 50,
    "ball_speed": [1, 2],
    'max_step': 100000,
    'lifes': 1
}



@click.command()
def main():
    # click.echo('Hello World!')
    # import Breakout
    # env = BreakoutS()
    # env = BreakoutSRA()
    # env = BreakoutSRAExt()
    # env = BreakoutNRA()
    # env.init(None)

    env = Breakout(conf)
    env = RewardAutomataWrapper(BreakoutDiscreteStateWrapper(env))

    # agent = RandomAgent([2, 3])
    # agent = SimpleAgent()
    # agent = NNAgent()
    # agent = SarsaAgent(env)

    # agent = RLAgent(env, RandomPolicyWithDecay(env, 2), QLearning(2))
    agent = RLAgent(env, RandomPolicyWithDecay(2), Sarsa(2))

    # filepath = "data/weights.p"
    filepath = "agent_data"

    render = False
    resume = False
    if resume:
        agent.load(filepath)

    if render:
        renderer = Renderer()

    stats = StatsManager()

    # Main training loop
    for ep in range(num_episodes):
        t = 0
        total_reward = 0

        state = env.reset()
        state, reward, done, info = env.step(1)

        while not done:
            action = agent.act(state)
            state2, reward, done, info = env.step(ID2ACTION[action])
            total_reward += reward

            agent.observe(state, action, reward, state2)
            agent.replay()

            state = state2

            if done:
                break

            if render:
                renderer.update(env.render())
                # import time
                # time.sleep(0.01)
            t += 1

        agent.reset()
        stats.update(len(agent.brain.Q), total_reward)
        stats.print_summary(ep, t, len(agent.brain.Q), total_reward, agent.exploration_policy.epsilon)
        if ep % 100 == 0:
            agent.save(filepath)

    agent.save(filepath)

    stats.plot()
    if render:
        renderer.release()


if __name__ == '__main__':
    main()
