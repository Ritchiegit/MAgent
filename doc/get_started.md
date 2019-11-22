# Get started
This tutorial will tell you the basic setting of the gridworld in MAgent, and show how to run the first demo.

## Environment
The basic environment is a large gridworld. Agents and walls can be added in the world.

## Agents
Agents are controlled by groups. Agents in a group share the same general attributes and control method.
The attributes of an agent can be width, height, speed, hp, etc. An AgentType can be registered as follows.

```python
predator = register_agent_type(
    "predator",
    {
        'width': 2, 'length': 2, 'hp': 1, 'speed': 1,
        'view_range': CircleRange(5), 'attack_range': CircleRange(2),
        'attack_penalty': -0.2
    })
```

## Observation
There are two parts in observation, spacial local view and non-spacial feature (see figure below).
- Spatial view consists of several rectangular channels. These channels will be masked by a circle or a sector as described in the agent type registration. (see view_range above).
  If the radius of a circle is 5, then the size of one channel is 11 x 11, where 11 = 5x2 + 1
- Non-spatial feature includes ID embedding, last action, last reward and normalized position.
  ID embedding is the binary representation of agent's unique ID.

空间视图由多个长方形通道组成。这些通道被一个圆或者一个扇形屏蔽。这一形状表示的是图形。

非空间特征包括ID 嵌入，上一个动作，上一次的回报，和正则化的位置。

Embedding ID是agent‘s特殊ID的表述。

<img src="../data/figure/observation_space.png" width="350">

The whole observation is an array of shape `(n_agents, view_width, view_height, n_channel)` for all agents. 

我理解的这个是整个屏幕的视野。

The above figure shows the observation for one agent. The spatial view contains 7 channels. `Wall` channel
is a "0/1" indicator to show whether there is a wall. `Group 1`, `Group 2` is the "0/1" indicators for agents in 
group 1, group 2. `Hp` is the normalized health point (range 0-1). `Minimap` is used to give a fuzzy global 
observation to the agents （no fog of war). （给每一个Agent一个全局的缩略图）The value in the minimap channel is computed as follows: 

1. Squeeze global map (e.g. 100x100) into minimap (e.g. 10x10) 放缩
2. The value of a minimap cell = (number of agents in this cell) / (number of all the agents) cell内agent的值就是放缩后对应块的值。



## Action
Actions are discrete actions. They can be move, attack and turn.
In the figure below, move range and attack range are also circular range (chunked*分块* to fit grids). 
The center point is the body point of the agent. Each point in the figure is a valid action.
So if agent is configured as follow, it has 13 + 8 + 2 = 23 valid actions.

<img src="../data/figure/action_space.png" width="300">

最后一个我认为是左右旋转，但左右旋转好像也没有什么用，因为attack和move都可以对一圈的范围都有效。

## Reward

Reward can be defined by constant*常量* attributes of agent type or by event trigger*事件触发的* .
Here is an example of the event tigger fashion. Boolean expression is supported. Two tigers can get reward when attack a deer simultaneously. 

```python
a = AgentSymbol(tiger_group, index='any')
b = AgentSymbol(tiger_group, index='any')
c = AgentSymbol(deer_group,  index='any')

# tigers get reward when they attack a deer simultaneously
e1 = Event(a, 'attack', c)
e2 = Event(b, 'attack', c)
config.add_reward_rule(e1 & e2, receiver=[a, b], value=[1, 1])
```
See [python/magent/builtin/config](../python/magent/builtin/config/) for more examples.
Of course, you can also write your own reward rules in the control logic in python code.

## Game loop & Model parallelism 并行性
In Magent, agents are controled by groups. You should use group handle to manipulate agents.

agent是成组被控制的，你可以通过一个groupd的操作杆来操纵agent

A typical main loop of a game is listed as follows. It is worth noting that groups can infer action in parallel.

一个组可以并行地计算下一个动作。

```python
handles = env.get_handles()
while not done:
    # take action for every group
    for i in range(n):
        obs[i] = env.get_observation(handles[i])
        ids[i] = env.get_agent_id(handles[i])
        # let models infer action in parallel (non-blocking)
        models[i].infer_action(obs[i], ids[i], block=False)  # 当前的状体其实就是 obs和ids model根据s返回一个action。

    for i in range(n):
        acts[i] = models[i].fetch_action()  # fetch actions (blocking) 
        env.set_action(handles[i], acts[i])  # 这里好像是环境因为agent的动作而发生改变
    
    done = env.step()
```
Also, you can train different groups in parallel.
```python
# train models in parallely
for i in range(n):
    models[i].train(print_every=1000, block=False)
for i in range(n):
    total_loss[i], value[i] = models[i].fetch_train()
```
Futhermore, with several lines of modification in python (use socket instead of pipe), models can be deployed to different machines.

在python中使用几行修改，使用socket而不是pipe。

## Run the first demo
Run the following command in the root directory, do not cd to `examples/`
```bash
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python examples/api_demo.py
```

In this environment, predators are pursuing preys. Predators can get rewards by attacking preys.
The predators and preys are trained by Deep Q-Network.
After training, predators learn to lock preys by cooperating with each other.  
See the source file [api_demo.py](../examples/api_demo.py) to know the basic api and workload of MAgent. 学习如何使用MAgent的Api
See also [train_pursuit.py](../examples/train_pursuit.py) to know how the above agents are trained. 学习agent是如何被训练的。

## Watch video
* Go to directory `build/render`
* Execute `./render`
* Open index.html in browser. A modal will be opened once the frontend gets connected to the backend 前端连接到后端，将打开模式（虽然输入进去只有[2019-11-19 23:07:16] [140301916313408] Listening on port 9030）
* Type `config.json` and `video_1.txt` in the two input boxes. 前一个是参数，后一个是一堆数字
* In the render, press arrow keys 'up', 'down', 'left', 'right' to move scope window. Press '<', '>' to zoom in or zoom out. Press 's' to adjust speed and progress. Press 'e' to re-input configuration file and map file.

## Play general-soldier game
In this section, Pygame are required.  
**Note for OSX user**: Unluckily, there is something wrong with Pygame on OSX, which makes this game very slow. You can skip this game if you are on OSX.

```base
pip install pygame
python examples/show_battle_game.py
```
In this game, you will act as a general and dispatch*调度* your soilders.
You have 10 chances to place your soilders in the map.
Then the soilders will act according to their deep q-networks.
You goal is to find best places to place your soilders and let them to eliminate the enemies.

## Next step
Try other examples and have fun!
