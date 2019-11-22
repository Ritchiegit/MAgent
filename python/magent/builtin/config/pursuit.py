import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    
    # 1. Register all kinds of agents.
    # 可以设置agents的相关属性 通过本函数注册所有的agent 类似于定义了一种agent为predator
    predator = cfg.register_agent_type(
        "predator",
        {
            'width': 2, 'length': 2, 'hp': 1, 'speed': 1,
            'view_range': gw.CircleRange(5), 'attack_range': gw.CircleRange(2),
            'attack_penalty': -0.2
        })
    # 2. 另一种agent为prey
    prey = cfg.register_agent_type(
        "prey",
        {
            'width': 1, 'length': 1, 'hp': 1, 'speed': 1.5,
            'view_range': gw.CircleRange(4), 'attack_range': gw.CircleRange(0)
        })

    # 3. Add all your agents to the environment through cfg.add_groups
    # 在这个文件里添加agents 通过 add_group函数添加agent
    predator_group  = cfg.add_group(predator)
    prey_group = cfg.add_group(prey)

    # 4. Construct an symbol for each kind of your agent
    # 构建agent的符号 添加了一个符号
    a = gw.AgentSymbol(predator_group, index='any')
    b = gw.AgentSymbol(prey_group, index='any')

    # design the reward rules
    # 设计Reward函数
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=[a, b], value=[1, -1])

    return cfg
