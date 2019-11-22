<center><b> A Tutotiral for Running Experiments with Many Kinds of Agents</b></center>

This tutorial will tell you how to modify the code to make it compatiable*相容* for running experiments with many kings of agents. In the following tutorial, we will take the *pursuit* environment as an example.
# Config Files
- Register all kinds of your agents through `cfg.regiser_agent_type` 通过本函数注册所有的agent
- Add all your agents to the environment through `cfg.add_groups` 在这个文件里添加agent
- Construct an symbol for each kind of your agent through `cfg.AgentSymbol ` 构建符号
- Finally, design the reward rules between all of your agents through `cfg.add_reward_rule` 设计reward rule

You could refer to the [**Pursuit.py**](https://github.com/geek-ai/MAgent/blob/master/python/magent/builtin/config/pursuit.py) to easily finish these steps. Here is one thing you need to notice, **when you do something wrong in modifying the config file like write a typo, the framework may just tell you that **`there is no such an environment`. So carefully check you code and do not believe the error message.

当你在修改配置文件出错时，框架只会告诉你：`there is no such an environment`

仔细检查你的code，不要相信出错信息。

# Training Code

There is mainly several parts you need to modify.

In the **main part of the training** code:

- Add all your agents' model into the list `models` 添加model
- Add all your agents' name into the list `names 添加names`
- Add all the evaluation arguments into the list `eval_obs` 添加评价参数

In the **generating agents part** of the code:

- Add all your kind of agents to the environment through `env.add_agents` (You may still get some incorrect error message in that step) 将所有agent添加到环境中。通过`env.add_agents`

You could also refer to the [**train_pursuit.py**](https://github.com/geek-ai/MAgent/blob/master/examples/train_pursuit.py) to understand these instructions better. Another tip is that do not use multi-processing training code if the number of your agents is too large, since the gc`garbage collection` of python is not quite well and it may lead you to the **OOM**.可能是 out of memory 或者out of mind): 

Have fun with our framework (: