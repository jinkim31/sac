import gym
from sac import SACagent


def main():
    max_episode_num = 200
    try:
        env = gym.make("Pendulum-v1")
    except BaseException:
        env = gym.make("Pendulum-v0")

    agent = SACagent(env)

    agent.train(max_episode_num)

    agent.plot_result()


if __name__ == "__main__":
    main()
