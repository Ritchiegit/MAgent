"""
Train a single model by self-play 自模拟
"""


import argparse
import time
import os
import logging as log
import math

import numpy as np

import magent


def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    # 初始化对战一开始屏幕左右两边的球
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    # 初始化的时候的左半圆handles[1]掌控的agent
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[0], method="custom", pos=pos)  # 添加agent

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    # 初始化的时候的右半圆handles[2]掌控的agent
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[1], method="custom", pos=pos)


def play_a_round(env, map_size, handles, models, print_every, train=True, render=False, eps=None):
    # TODO_ 这个是怎么可视化的 show_battle_game.py 可以显示
    env.reset()
    generate_map(env, map_size, handles)  # 调用上面的函数初始化两团agent

    step_ct = 0
    done = False

    # 存放obs ids act reward
    n = len(handles)
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    # 清空缓冲区以存储agent的所有记录。 一个agent一个入口
    sample_buffer = magent.utility.EpisodesBuffer(capacity=1500)  # 比train_battle 多一行
    total_reward = [0 for _ in range(n)]

    print("===== sample =====")
    print("eps %.2f number %s" % (eps, nums))
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])  # TODO 3 得到一类agent观察到的值 观察到了什么样的世界
            ids[i] = env.get_agent_id(handles[i])
            acts[i] = models[i].infer_action(obs[i], ids[i], 'e_greedy', eps=eps)  # TODO 1 这个就是agent的action　包括算法、attack
            # 这里直接对env进行修改我认为有问题，因为原本t时刻的action仅由t-1时刻的状态决定
            # 但如果这样的话，就会使得前面先被更新的agent的行为可以被后更新的agent观察到
            # 故，不符合对t-1时间的依赖 PS.先被更新的更吃亏

            env.set_action(handles[i], acts[i])  # 将action作用于env　# TODO 2 这里可能会对agent 进行一些攻击等
            # 我认为应该将上面这行更改为如下三行
        """
        # 可以参考train_battle.py
        for i in range(n):
            acts[i] = models[i].fetch_action()  # fetch actions (blocking)  # 将每个智能体的行为添加到环境中
            env.set_action(handles[i], acts[i])
        """


        # simulate one step
        done = env.step()
        # 执行单个步骤，观察是否结束

        # sample
        step_reward = []
        for i in range(n):
            rewards = env.get_reward(handles[i])
            # TODO_ 2 可能会受伤，根据造成的伤害和自己受伤得到 Reward
            # 还是使用c函数实现的
            if train:
                alives = env.get_alive(handles[i])
                # TODO_ 这个在pursuit中没有遇到过 sample_buffer.record_step
                sample_buffer.record_step(ids[i], obs[i], acts[i], rewards, alives)
            s = sum(rewards)
            step_reward.append(s)  # TODO_ 这里是使用梯度下降吗 只是用于展示
            total_reward[i] += s

        # render
        if render:
            env.render()

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        # clear dead agents
        env.clear_dead()  # TODO_ 这里是在哪里受伤呢？ 是在上面的 set_env中

        # 这里少了几行 if args.train: blabla model check_done()

        if step_ct % print_every == 0:
            print("step %3d,  nums: %s reward: %s,  total_reward: %s " %
                  (step_ct, nums, np.around(step_reward, 2), np.around(total_reward, 2)))
        step_ct += 1
        if step_ct > 550:
            break

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # sample_buffer 包含 buffer, capacity, is_full

    # train
    total_loss, value = 0, 0
    if train:
        print("===== train =====")
        start_time = time.time()
        total_loss, value = models[0].train(sample_buffer, 1000)  # 这里是通过sample_buffer 将训练信息传过来
        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    def round_list(l): return [round(x, 2) for x in l]
    return total_loss, nums, round_list(total_reward), value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every",     type=int, default=5)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=2000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=125)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="battle")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument('--alg', default='dqn', choices=['dqn', 'drqn'])
    args = parser.parse_args()

    # 初始化被写在 magent.utility 中 init_logger() 函数中了。
    # set logger
    log.basicConfig(level=log.INFO, filename=args.name + '.log')
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    log.getLogger('').addHandler(console)

    # init the game
    env = magent.GridWorld("battle", map_size=args.map_size)
    env.set_render_dir("build/render")

    # two groups of agents
    handles = env.get_handles()

    # sample eval observation set
    eval_obs = None
    if args.eval:
        print("sample eval set...")
        env.reset()
        generate_map(env, args.map_size, handles)
        eval_obs = magent.utility.sample_observation(env, handles, 2048, 500)[0]

    # init models
    batch_size = 512
    unroll_step = 8
    target_update = 1200
    train_freq = 5

    models = []
    if args.alg == 'dqn':
        from magent.builtin.tf_model import DeepQNetwork
        models.append(DeepQNetwork(env, handles[0], args.name,  # 这里多加了本行的参数
                                   batch_size=batch_size,
                                   learning_rate=3e-4,
                                   memory_size=2 ** 21, target_update=target_update,
                                   train_freq=train_freq, eval_obs=eval_obs))
    elif args.alg == 'drqn':
        from magent.builtin.tf_model import DeepRecurrentQNetwork
        models.append(DeepRecurrentQNetwork(env, handles[0], args.name,
                                   learning_rate=3e-4,
                                   batch_size=batch_size/unroll_step, unroll_step=unroll_step,
                                   memory_size=2 * 8 * 625, target_update=target_update,
                                   train_freq=train_freq, eval_obs=eval_obs))
    else:
        # see train_against.py to know how to use a2c
        raise NotImplementedError

    models.append(models[0])

    # load if
    savedir = 'save_model'
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)
    else:
        start_from = 0

    # print debug info
    print(args)
    print("view_space", env.get_view_space(handles[0]))
    print("feature_space", env.get_feature_space(handles[0]))

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = magent.utility.piecewise_decay(k, [0, 700, 1400], [1, 0.2, 0.05]) if not args.greedy else 0
        # 下面就开始训练啦
        loss, num, reward, value = play_a_round(env, args.map_size, handles, models,
                                                train=args.train, print_every=50,
                                                render=args.render or (k+1) % args.render_every == 0,
                                                eps=eps)  # for e-greedy

        log.info("round %d\t loss: %s\t num: %s\t reward: %s\t value: %s" % (k, loss, num, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        # save models
        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            for model in models:
                model.save(savedir, k)
