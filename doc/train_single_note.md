# TODO 0 训练好的结果是怎么可视化的

解读 show_battle_game.py

check_model 根据参数检查是否包含对应的model，如果没有对应的model则去下载

保证有这个模型之后，就可以调用render对模型进行绘制

`PyGameRenderer().start(Server())`中PyGameRenderer()是Pygame的一个渲染和函数，和Battle无关

BattleServer 是绘制这个场景的核心，确实一个一直运动的场景肯定需要后台

gw.Config会包括config_dict agent_type_dict groups reward_rules 应该是我们需要重点设置的对象

# TODO 1 修改agent可以进行的攻击，eg. 开炮

更改攻击方式 修改cfg.register_agent_type


# TODO 2 修改agent受攻击之后的结果，可能是在set_action处更改，也有可能是在 Reward_step中更改

add_reward_rule设置进行反馈的方式

# TODO 3 get_observation(handles[1]) 观察到了怎样的环境，如何对其进行修改

更改observation方式 修改cfg.register_agent_type

# TODO 这个在pursuit中没有遇到过 sample_buffer.record_step

# TODO step_reward.append(s)  # TODO 这里是使用梯度下降吗