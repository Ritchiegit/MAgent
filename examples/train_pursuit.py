"""
Pursuit: predators get reward when they attack prey.
"""

import argparse
import time
import logging as log

import numpy as np

import magent
from magent.builtin.tf_model import DeepQNetwork


def play_a_round(env, map_size, handles, models, print_every, train=True, render=False, eps=None):
    # 进行单轮 env map handles
    env.reset()
    
    # 添加模型 添加agents 和 walls
    env.add_walls(method="random", n=map_size * map_size * 0.03)
    env.add_agents(handles[0], method="random", n=map_size * map_size * 0.0125)
    env.add_agents(handles[1], method="random", n=map_size * map_size * 0.025)

    # 没有中间设置DQN等模型的过程 也没有载入model的过程

    step_ct = 0  # TODO
    done = False

    n = len(handles)
    obs  = [[] for _ in range(n)]  # n 个空列表的 列表
    ids  = [[] for _ in range(n)]  # 用于存放当前Agent观察到的环境
    acts = [[] for _ in range(n)]  # 用于存放当前Agent的ID
    nums = [env.get_num(handle) for handle in handles]  # 获得Agent的数量
    total_reward = [0 for _ in range(n)]  # 对于每个agent的reward

    print("===== sample =====")
    print("eps %s number %s" % (eps, nums))  # TODO what is eps?
    start_time = time.time()
    while not done:  # 一步
        # take actions for every model  每个Agent都进行返回
        for i in range(n):
            obs[i] = env.get_observation(handles[i])  # 观察到环境
            ids[i] = env.get_agent_id(handles[i])  # 当前agent的标号
            # let models infer action in parallel (non-blocking)
            models[i].infer_action(obs[i], ids[i], 'e_greedy', eps, block=False)  #　推断出来结果
        for i in range(n):
            acts[i] = models[i].fetch_action()  # fetch actions (blocking)  # 将每个智能体的行为添加到环境中
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample  感觉是计算活着的reward
        # 这里计算单个agent收获的总的reward
        step_reward = []
        for i in range(n):
            # TODO env.get_reward
            rewards = env.get_reward(handles[i])  # 根据规则获得reward 当每个agent都采取行动后，给出其reward
            if train:
                alives  = env.get_alive(handles[i])  # env.get_alive 一个Agent是否存活
                # store samples in replay buffer (non-blocking)
                # 这个在api里没有提到
                models[i].sample_step(rewards, alives, block=False)  # TODO models[i].sample_step 记录单步
            s = sum(rewards)  # 计数单个agent的 总reward
            step_reward.append(s)  # 记录下来 reward
            total_reward[i] += s

        # render
        if render:
            env.render()  # 这个应该是绘制渲染

        # clear dead agents
        env.clear_dead()  # 清除掉死亡的agent

        # check 'done' returned by 'sample' command
        if train:
            for model in models:
                model.check_done()

        if step_ct % print_every == 0:
            print("step %3d,  reward: %s,  total_reward: %s " %
                  (step_ct, np.around(step_reward, 2), np.around(total_reward, 2)))
        step_ct += 1
        if step_ct > 250:
            break

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # train
    total_loss, value = [0 for _ in range(n)], [0 for _ in range(n)]
    if train:
        print("===== train =====")
        start_time = time.time()

        # train models in parallel
        for i in range(n):
            models[i].train(print_every=2000, block=False)
        for i in range(n):
            total_loss[i], value[i] = models[i].fetch_train()

        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    return magent.round(total_loss), magent.round(total_reward), magent.round(value)


if __name__ == "__main__":
    # http://wiki.jikexueyuan.com/project/explore-python/Standard-Modules/argparse.html
    parser = argparse.ArgumentParser()  # Python 内置的一个用于命令项选项与参数解析的模块。
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=1000)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--name", type=str, default="pursuit")
    args = parser.parse_args()

    # set logger  初始化logs
    magent.utility.init_logger(args.name)

    # init the game  初始化环境并进行渲染
    env = magent.GridWorld("pursuit", map_size=args.map_size)
    env.set_render_dir("build/render")

    # two groups of agents  给agents添加两个柄
    handles = env.get_handles()

    # load models
    names = ["predator", "prey"]
    models = []

    for i in range(len(names)):
        models.append(magent.ProcessingModel(  # 载入一个预定义的Model
            env, handles[i], names[i], 20000+i, 4000, DeepQNetwork,  # 载入env，handle，name
            batch_size=512, memory_size=2 ** 22,
            target_update=1000, train_freq=4
        ))

    # load if
    savedir = 'save_model'  # 存储在这一 文件夹下
    if args.load_from is not None:  # load_from读取数据
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)  # 读取已经生成的model
    else:
        start_from = 0  # 不然就从位置0开始读入数据

    # print debug info
    print(args)
    print("view_space", env.get_view_space(handles[0]))  # 可视化区域
    print("feature_space", env.get_feature_space(handles[0]))  # TODO

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        # 分片衰变
        eps = magent.utility.piecewise_decay(k, [0, 200, 400], [1, 0.2, 0.05]) if not args.greedy else 0

        loss, reward, value = play_a_round(env, args.map_size, handles, models,
                                           print_every=50, train=args.train,
                                           render=args.render or (k+1) % args.render_every == 0,
                                           eps=eps)  # for e-greedy
        log.info("round %d\t loss: %s\t reward: %s\t value: %s" % (k, loss, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            for model in models:
                model.save(savedir, k)

    # send quit command
    for model in models:
        model.quit()
