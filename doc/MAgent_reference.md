### MAgent简介

ACI: Artificial Collective Intelligence 集群人工智能

多智能体强化学习平台。

因为使用了共享网络network sharing和ID embeddingMAgent is 高度可拓展的，并且可以在单GPU上运行1000,000 个Agent

并且，MAgent 提供环境和Agent的配置文件，并且一个reward的描述，达到一个灵活的环境设计。

最终，MAgent 会提供一个简单但视觉效果很好的渲染，用以迭代地表现环境和agent的状态。

User 能够滑动或者放缩区域，甚至可以在game中操作agent



### Gridworld Fundamentals

网格结构是这大量agent的基本环境。

单个agent可以是一个长方体，都含有局部的细节信息和可选的全局信息。

其动作可以是移动、转向、攻击。

一个c++的引擎用于支持完成这一快速模拟。

多种agent都可以在上面运行。

在状态空间中，action空间和agent属性可以被定义，不同的环境可以被设计出来。



### Reward Description Language

在我们的接口中，用户可以描述reward，agent symbols，甚至一些event事件。

（我猜想同时添加多个agent到途中可能就是一个event）

when the boolean expression of an event expression is true, reward will be assigned to the agents involved in the events

当event表达是一个true时，reward会分配给事件中涉及到的agent。



### Live and Interactive Part

DQN、DRQN、A2C在该平台上可用。



还保留了用户的接口

因此在我们的demo中可以将agent放在某一位置。







## 随想

也许两个网络可以达到更好的效果，其实在第一次玩的时候，没有达到战无不胜的效果。（可能对agent的训练到位了，但在哪个位置释放agent没有到位。



## 该怎么做

不只是死磕这一篇论文，也可以找找相关的文章

跑跑代码，找找思路。



引用这一篇论文的所有文章

https://arxiv.org/abs/1712.00600v1

张伟楠

https://github.com/cityflow-project/CityFlow



