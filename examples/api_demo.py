"""
First demo, show the usage of API
"""

import magent
# try:
#     from magent.builtin.mx_model import DeepQNetwork
# except ImportError as e:
from magent.builtin.tf_model import DeepQNetwork

if __name__ == "__main__":
    map_size = 100

    # init the game "pursuit"  (config file are stored in python/magent/builtin/config/)
    # 初始化环境
    env = magent.GridWorld("pursuit", map_size=map_size)
    env.set_render_dir("build/render")  # 设置 render目录

    # get group handles
    predator, prey = env.get_handles()  # 获得group的handle

    # init env and agents 添加model
    env.reset()
    env.add_walls(method="random", n=map_size * map_size * 0.01)
    env.add_agents(predator, method="random", n=map_size * map_size * 0.02)
    env.add_agents(prey,     method="random", n=map_size * map_size * 0.02)

    # init two models 初始化两个model
    model1 = DeepQNetwork(env, predator, "predator")
    model2 = DeepQNetwork(env, prey,     "prey")

    # load trained model  载入训练过的model
    model1.load("data/pursuit_model")
    model2.load("data/pursuit_model")

    done = False
    step_ct = 0
    print("nums: %d vs %d" % (env.get_num(predator), env.get_num(prey)))
    while not done:
        # take actions for deers  # 对每个agent都返回obs ids，和对应的action
        obs_1 = env.get_observation(predator)
        ids_1 = env.get_agent_id(predator)
        acts_1 = model1.infer_action(obs_1, ids_1)
        env.set_action(predator, acts_1)  # agent的action影响环境

        # take actions for tigers
        obs_2  = env.get_observation(prey)
        ids_2  = env.get_agent_id(prey)
        acts_2 = model2.infer_action(obs_2, ids_1)
        env.set_action(prey, acts_2)

        # simulate one step
        done = env.step()  # 模拟单步 的操作

        # render
        env.render()  # 进行渲染

        # get reward 计算单个agent收获的总reward
        reward = [sum(env.get_reward(predator)), sum(env.get_reward(prey))]

        # clear dead agents
        env.clear_dead()

        # print info
        if step_ct % 10 == 0:
            print("step: %d\t predators' reward: %d\t preys' reward: %d" %
                    (step_ct, reward[0], reward[1]))

        step_ct += 1
        if step_ct > 250:
            break

