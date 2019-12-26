# TODO 0 训练好的结果是怎么可视化的

解读 show_battle_game.py

check_model 根据参数检查是否包含对应的model，如果没有对应的model则去下载

保证有这个模型之后，就可以调用render对模型进行绘制

`PyGameRenderer().start(Server())`中PyGameRenderer()是Pygame的一个渲染和函数，和Battle无关

BattleServer 是绘制这个场景的核心，确实一个一直运动的场景肯定需要后台

训练好的参数是怎么进去的呢？在哪里读入参数和训练好的模型的呢?

## 参数读入

/home/rc/Python/MAgent/python/magent/gridworld.py

line 12 `load_config` 读入各种配置 `class Config`中所有配置。

## 模型读入

/home/rc/Python/MAgent/python/magent/renderer/server/battle_server.py

line 105

载入训练好的DeepQNetwork

```python
		models = []
        models.append(DeepQNetwork(env, handles[0], 'trusty-battle-game-l', use_conv=True))  # 添加训练完的model
        models.append(DeepQNetwork(env, handles[1], 'trusty-battle-game-r', use_conv=True))

        # load model
        models[0].load(path, 0, 'trusty-battle-game-l')  # 这个和上边可能共同起作用吧。
        models[1].load(path, 0, 'trusty-battle-game-r')

```

但是这个 'trusty-battle-game'是在哪里存入的呢？



​	





gw.Config会包括config_dict agent_type_dict groups reward_rules 应该是我们需要重点设置的对象

# TODO 1 修改agent可以进行的攻击，eg. 开炮

更改攻击方式 修改cfg.register_agent_type

## 如何设定agent的参数

MAgent/python/magent/gridworld.py line 578

```python
class Config:
    # 整个类都是在设定场景、agent、reward的参数。
	def __init__(self):
		self.config_dict = {}
		self.agent_type_dict = {}
		self.groups = []
		self.reward_rules = []
    def set(self, args):
        # 将环境参数依次复制过来
    def register_agent_type(self, name, attr):
        # 将agent参数依次复制过来
        """
        name
        attr: dict —— agent type 用一个字典存储
        height
        width
        speed
        hp
        view_range: gw.圆形/ gw.扇形
        
        damage:  attack damage
        step_recover: 每一步的生命恢复值 可以是负数
        kill_supply: 杀死其可以获得多少hp
        
        step_reward: 每一步的回报
        kill_reward: 杀死其的agent可以获得多少reward
        dead_penalty: 死亡的reward
        attack_penalty: 攻击的reward 可以是负数
        
        这里没有说攻击范围可能 可能与view_range相同，仔细看看其中部分
        """
```

没有说攻击范围，仔细看看view_range 部分 TODO

battle_single 是在哪里设置的

```python
	def add_reward_rule(self, on, receiver,  value, terminal=False):
        # 添加reward 
        # 若接受方不是决定性代理，它必须是触发事件中涉及的代理之一。 这个意思是reward不能由两个不重要的agent触发。
        """
        on :trigger event 的布尔型表达 ？？？
        receiver 那些类型的agent受reward rule的限制
        value 需要赋值的值
        terminal 改event 是否会终止整场游戏
        """
        
```

## 参数在哪里设定？

/home/rc/Python/MAgent/python/magent/renderer/server/battle_server.py line 12

`def load_config`**载入各种设定**。
agent 参数
环境 参数
添加agent group，设定agent的symbol
添加reward rule

```python
def load_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()  # 这里会包括config_dict agent_type_dict groups reward_rules 应该是我们需要重点设置的对象

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})

    cfg.set({"embedding_size": 10})
	# 这里的参数需要和训练时候的参数纬度对应起来，否则会纬度不匹配。InvalidArgumentError (see above for traceback): Assign requires shapes of both tensors to match. lhs shape= [512,13] rhs shape= [512,21]
    # 这里是设置了agent
    small = cfg.register_agent_type(  # 这里可以直接设置agent的类型
        "small",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),  # 观察到的区域，攻击区域
         'damage': 2, 'step_recover': 0.1,
         'step_reward': -0.001, 'kill_reward': 100, 'dead_penalty': -0.05, 'attack_penalty': -1,
         })

    # 添加 左右两组group
    g0 = cfg.add_group(small)
    g1 = cfg.add_group(small)

    # 设置Symbol
    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    # 添加reward_rule
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=2)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=2)

    return cfg
```

这里面的 add_reward_rule 需要细看 
（直接赋值，再细看看`gw.Event(a, 'attack', b)`）

```
	node.op = EventNode.OP_ATTACK
	node.inputs = [subject, args[0]]
```

`cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=2)`将node 加入环境，但这个具体在哪里利用呢？

如何应用这个rule呢？

## 但这些参数设定完之后，这些规则，在哪里怎么被应用呢？TODO

## BattleServer 的设定和使用

```python
# 初始化的时候设定
map_size
env
eps
handles
models


# 定义时传入
total_step
add_interval 
add_counter
done

class BattleServer(BaseServer):
    def __init__(self, ...):
        pass
    
    def step(self):
        # 模拟单步 
        # 用于model[i].infer_action 和 env.set_action() 之后进行单步模拟
        obs = ...
        ids = ...
        
        for i in range(len(handles)):
            act =
            env.set_action
            # 这里多了一个counter
            # env.get_action_space() 是用来获得action 空间，并计数，但这里 变成0了，只取个形状
            couter.append(np.zeros(shape=env.get_action_space(handles[i])))
            for j in acts:
                counter[-1][j] += 1
            # env.step 这里是调用c程序中的 step
            done = env.step()  # _LIB.env_step(self.game, ctypes.byref(done)) 想要实验这行代码，命令行直接 core dumped
            env.clear_dead()
           """
        	>>> path_to_sofile = "/home/rc/Python/MAgent/build/libmagent.so"
            >>> lib = ctypes.CDLL(path_to_sofile, ctypes.RTLD_GLOBAL)
            >>> lib
            <CDLL '/home/rc/Python/MAgent/build/libmagent.so', handle 56470ebf8620 at 0x7fcc87fe2080>
            >>> lib.env_step()
            [2]    4680 segmentation fault (core dumped)  python
			"""
    def get_data(self, frame_id, y_range):
        # 获得agent的 信息，event的信息
        self.done = self.step()
        pos, event  = self.env._get_render_info(x_range, y_range)
        return pos, event
    def add_agents(self, x, y, g)
    	pos = ...
        self.env.add.agents(self.handles[g], method="custom", pos)
    def get_banners(self, frame_id, resolution)
    	# 标题
        pass
    
```









# 用于agent设置的东西

scene 场景中的墙，如何设置，一个元素一个元素分吗?

​	看看有没有生成随机岛屿的方法

reward 其实也就是杀与被杀 走路的reward

HP

damage

---

get observation

attack TODO








## TODO 2 修改agent受攻击之后的结果，可能是在set_action处更改，也有可能是在 Reward_step中更改

add_reward_rule 设置进行反馈的方式

set_action 在环境中的应用

## TODO 3 get_observation(handles[1]) 观察到了怎样的环境，如何对其进行修改

更改observation方式 修改cfg.register_agent_type

## TODO step_reward.append(s)  # TODO 

这里是将reward添加进去，便于展示，最终用于梯度下降的还是刚才在buffer里记录的东西。

