参考教程

https://blog.csdn.net/weixin_39916966/article/details/88758693

https://blog.csdn.net/CAU_Ayao/article/details/83627342

https://developer.nvidia.com/cuda-92-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu

https://blog.csdn.net/Aiolia86/article/details/80342240

https://blog.csdn.net/qq_33838170/article/details/83217880



整体参考

https://blog.csdn.net/weixin_39916966/article/details/88758693





## 安装驱动

点左下角的应用找到 软件和更新

将“Ubuntu软件”中“下载自”更新成阿里源_http://mirrors.aliyun.com/ubuntu

点击“附加驱动”，选nvidia-driver-390的驱动





## Cudnn

https://developer.nvidia.com/rdp/cudnn-download

点开 Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 9.0]，点击

[cuDNN Library for Linux](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/9.0_20191031/cudnn-9.0-linux-x64-v7.6.5.32.tgz) 下载



生成cuDNN软连接

https://blog.csdn.net/xierhacker/article/details/53035989

(base) XXX:/usr/local/cuda/lib64$ sudo ln -s libcudnn.so.7.6.5 libcudnn.so.7
(base) XXX:/usr/local/cuda/lib64$ sudo ln -s libcudnn.so.7 libcudnn.so
(base) XXX:/usr/local/cuda/lib64$ sudo ldconfig -v

要跟自己刚复制进去的cudnn版本相对应，可以进到/usr/local/cuda/lib64 直接查看

测试cudnn是否可用 这个解决了的问题

```
Cublas failure
Error code 1
mnistCUDNN.cpp:404
```

https://blog.csdn.net/zbbmm/article/details/102680387

为了编译，需要先g++ gcc 降级

```
sudo apt install gcc-5
sudo apt install g++-5

sudo mv gcc gcc.bak #备份
sudo ln -s gcc-5 gcc #重新链接gcc
sudo mv g++ g++.bak #备份
sudo ln -s g++-5 g++　#重新链接g++
```
出现了


```
CUDNN failure
Error: CUDNN_STATUS_INTERNAL_ERROR
mnistCUDNN.cpp:394
Aborting...
```
解决说nvidia cacha被污染了

`sudo rm -rf ~/.nv/`

这里，我直接运行这行代码报错不变，关掉所有浏览器、窗口

https://www.alatortsev.com/2018/01/17/fixing-cudnn_status_internal_error/

```shell
(base) rc@rc-pc:~/Python/cudnntest/mnistCUDNN$ sudo rm -rf ~/.nv/
(base) rc@rc-pc:~/Python/cudnntest/mnistCUDNN$ ./mnistCUDNN
cudnnGetVersion() : 7605 , CUDNN_VERSION from cudnn.h : 7605 (7.6.5)
Host compiler version : GCC 5.5.0
There are 1 CUDA capable devices on your machine :
device 0 : sms  3  Capabilities 5.0, SmClock 1241.5 Mhz, MemSize (Mb) 2004, MemClock 900.0 Mhz, Ecc=0, boardGroupID=0
Using device 0

Testing single precision
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 1
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.076928 time requiring 3464 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.081632 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.115584 time requiring 57600 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.313024 time requiring 207360 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.566432 time requiring 2057744 memory
Resulting weights from Softmax:
0.0000000 0.9999399 0.0000000 0.0000000 0.0000561 0.0000000 0.0000012 0.0000017 0.0000010 0.0000000 
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 0.9999288 0.0000000 0.0000711 0.0000000 0.0000000 0.0000000 0.0000000 
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 0.9999820 0.0000154 0.0000000 0.0000012 0.0000006 

Result of classification: 1 3 5

Test passed!

Testing half precision (math in single precision)
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 1
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.030976 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.039488 time requiring 3464 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 0.058848 time requiring 28800 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.203840 time requiring 207360 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.533696 time requiring 2057744 memory
Resulting weights from Softmax:
0.0000001 1.0000000 0.0000001 0.0000000 0.0000563 0.0000001 0.0000012 0.0000017 0.0000010 0.0000001 
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 1.0000000 0.0000000 0.0000714 0.0000000 0.0000000 0.0000000 0.0000000 
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 1.0000000 0.0000154 0.0000000 0.0000012 0.0000006 

Result of classification: 1 3 5

Test passed!

```




## 安装MiniConda

https://www.jianshu.com/p/fab0068a32b4

创建虚拟环境

https://blog.csdn.net/a493823882/article/details/87888509

给环境重命名的方法

https://www.jianshu.com/p/7265011ba3f2



Oh my zsh

https://ohmyz.sh/

https://www.linuxidc.com/Linux/2017-10/147755.htm

https://www.zhihu.com/question/33277508



参考环境

```shell
$ uname -a                     
Linux rc-pc 5.0.0-37-generic #40~18.04.1-Ubuntu SMP Thu Nov 14 12:06:39 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux
$ cmake --version
cmake version 3.14.0
CMake suite maintained and supported by Kitware (kitware.com/cmake).
$ g++ --version                
g++ (Ubuntu 5.5.0-12ubuntu1) 5.5.0 20171010

```

