# 炸内存

```shell
2019-12-10 20:43:21.353944: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1084] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1509 MB memory) -> physical GPU (device: 0, name: GeForce 940MX, pci bus id: 0000:01:00.0, compute capability: 5.0)
Process Process-1:
Traceback (most recent call last):
  File "/home/rc/miniconda3/envs/withtensorflow19/lib/python3.6/multiprocessing/process.py", line 258, in _bootstrap
    self.run()
  File "/home/rc/miniconda3/envs/withtensorflow19/lib/python3.6/multiprocessing/process.py", line 93, in run
    self._target(*self._args, **self._kwargs)
  File "/home/rc/Python/MAgent/python/magent/model.py", line 303, in model_client
    model = RLModel(**model_args)
  File "/home/rc/Python/MAgent/python/magent/builtin/tf_model/dqn.py", line 143, in __init__
    self.replay_buf_view     = ReplayBuffer(shape=(memory_size,) + self.view_space)
  File "/home/rc/Python/MAgent/python/magent/builtin/common.py", line 9, in __init__
    self.buffer = np.empty(shape=shape, dtype=dtype)
MemoryError: Unable to allocate array with shape (2097152, 13, 13, 7) and data type float32
```

这一个变量 9 G 谁顶得住啊...

2097152\*13\*13\*7个float

2097152\*13\*13\*7\*4Byte

= 9,923,723,264 Byte

用服务器吧



# libcufft.so.8.0.61 被污染

```shelll
The following NEW packages will be INSTALLED:

  bleach             anaconda/cloud/conda-forge/linux-64::bleach-1.5.0-py36_0
  cudatoolkit        anaconda/pkgs/free/linux-64::cudatoolkit-8.0-3
  cudnn              anaconda/pkgs/main/linux-64::cudnn-7.1.3-cuda8.0_0
  html5lib           anaconda/cloud/conda-forge/linux-64::html5lib-0.9999999-py36_0
  libprotobuf        anaconda/cloud/conda-forge/linux-64::libprotobuf-3.11.1-h8b12597_0
  markdown           anaconda/cloud/conda-forge/noarch::markdown-3.1.1-py_0
  protobuf           anaconda/cloud/conda-forge/linux-64::protobuf-3.11.1-py36he1b5a44_0
  tensorflow-gpu     anaconda/pkgs/main/linux-64::tensorflow-gpu-1.4.1-0
  tensorflow-gpu-ba~ anaconda/pkgs/main/linux-64::tensorflow-gpu-base-1.4.1-py36h01caf0a_0
  tensorflow-tensor~ anaconda/pkgs/main/linux-64::tensorflow-tensorboard-1.5.1-py36hf484d3e_1
  webencodings       anaconda/cloud/conda-forge/noarch::webencodings-0.5.1-py_1
  werkzeug           anaconda/cloud/conda-forge/noarch::werkzeug-0.16.0-py_0


Proceed ([y]/n)? y

Preparing transaction: done
Verifying transaction: | 
SafetyError: The package for cudatoolkit located at /home/lqz/anaconda3/pkgs/cudatoolkit-8.0-3
appears to be corrupted. The path 'lib/libcufft.so.8.0.61'
has an incorrect size.
  reported size: 146772120 bytes
  actual size: 107263488 bytes

ClobberError: The package 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main::cudnn-7.1.3-cuda8.0_0' cannot be installed due to a path collision for 'include/cudnn.h'. This path already exists in the target prefix, and it won't be removed by an uninstall action in this transaction. The path is one that conda doesn't recognize. It may have been created by another package manager.


/ 
done
Executing transaction: done

```

删除/home/lqz/anaconda3/pkgs/中 cudatoolkit-8.0-3整个文件夹

就没有lib/libcufft.so.8.0.61被污染的问题了

再重新安装 tensorflow-gpu 1.4.1

`conda uninstall tensorflow-gpu`

`conda install tensorflow-gpu==1.4.1`

下面是检查哪些gpu可用。

验证下

```shell
# https://blog.csdn.net/u013538542/article/details/89008638
>>> from tensorflow.python.client import device_lib
>>> 
>>> def get_available_gpus():
...     local_device_protos = device_lib.list_local_devices()
...     return [x.name for x in local_device_protos if x.device_type == 'GPU']
... 

>>> get_available_gpus()
2019-12-11 12:05:49.796875: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
2019-12-11 12:05:50.321464: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 

name
...

2019-12-11 12:05:51.105083: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Device peer to peer matrix
2019-12-11 12:05:51.105141: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1051] DMA: 0 1 2 
2019-12-11 12:05:51.105155: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 0:   Y N N 
2019-12-11 12:05:51.105171: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 1:   N Y N 
2019-12-11 12:05:51.105189: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 2:   N N Y 
2019-12-11 12:05:51.105212: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: TeslaP40, pci bus id: 0000:3b:00.0, compute capability: 6.1)
2019-12-11 12:05:51.105229: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:1) -> (device: 1, name: Tesla P40, pci bus id: 0000:86:00.0, compute capability: 6.1)
2019-12-11 12:05:51.105242: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:2) -> (device: 2, name: Tesla P40, pci bus id: 0000:af:00.0, compute capability: 6.1)
['/device:GPU:0', '/device:GPU:1', '/device:GPU:2']

```





# Loaded runtime CuDNN library: 

## 7103 (compatibility version 7100) but source was compiled with 7005 (compatibility version 7000)

```shell
2019-12-11 13:56:50.504534: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Device peer to peer matrix
2019-12-11 13:56:50.504601: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1051] DMA: 0 1 2 
2019-12-11 13:56:50.504618: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 0:   Y Y Y 
2019-12-11 13:56:50.504741: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 1:   Y Y Y 
2019-12-11 13:56:50.504753: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 2:   Y Y Y 
2019-12-11 13:56:50.504783: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Tesla P40, pci bus id: 0000:3b:00.0, compute capability: 6.1)
2019-12-11 13:56:50.504803: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:1) -> (device: 1, name: Tesla P40, pci bus id: 0000:86:00.0, compute capability: 6.1)
2019-12-11 13:56:50.504824: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:2) -> (device: 2, name: Tesla P40, pci bus id: 0000:af:00.0, compute capability: 6.1)
Namespace(alg='dqn', eval=False, greedy=False, load_from=None, map_size=125, n_round=2000, name='battle', render=False, render_every=10, save_every=5, train=True)
view_space (13, 13, 7)
feature_space (34,)
===== sample =====
eps 1.00 number [625, 625]
2019-12-11 13:56:54.126921: E tensorflow/stream_executor/cuda/cuda_dnn.cc:378] Loaded runtime CuDNN library: 7103 (compatibility version 7100) but source was compiled with 7005 (compatibility version 7000).  If using a binary install, upgrade your CuDNN library to match.  If building from sources, make sure the library loaded at runtime matches a compatible version specified during compile configuration.
2019-12-11 13:56:54.127566: E tensorflow/stream_executor/cuda/cuda_dnn.cc:378] Loaded runtime CuDNN library: 7103 (compatibility version 7100) but source was compiled with 7005 (compatibility version 7000).  If using a binary install, upgrade your CuDNN library to match.  If building from sources, make sure the library loaded at runtime matches a compatible version specified during compile configuration.
2019-12-11 13:56:54.128554: F tensorflow/core/kernels/conv_ops.cc:667] Check failed: stream->parent()->GetConvolveAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms) 
2019-12-11 13:56:54.129174: F tensorflow/core/kernels/conv_ops.cc:667] Check failed: stream->parent()->GetConvolveAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms) 
Traceback (most recent call last):
  File "examples/train_battle.py", line 221, in <module>
    eps=eps)  # for e-greedy
  File "examples/train_battle.py", line 70, in play_a_round
    acts[i] = models[i].fetch_action()  # fetch actions (blocking)
  File "/home/lqz/MAgent/python/magent/model.py", line 211, in fetch_action
    info = self.conn.recv()
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/multiprocessing/connection.py", line 407, in _recv_bytes
    buf = self._recv(4)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/multiprocessing/connection.py", line 383, in _recv
    raise EOFError
EOFError




(withtensorflow15) [lqz@gpu03 MAgent]$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Tue_Jan_10_13:22:03_CST_2017
Cuda compilation tools, release 8.0, V8.0.61
(withtensorflow15) [lqz@gpu03 MAgent]$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
#define CUDNN_MAJOR 7
#define CUDNN_MINOR 1
#define CUDNN_PATCHLEVEL 4

```

https://docs.nvidia.com/deeplearning/sdk/cudnn-install/

解决方案 将cudnn 换成 7.0.5

https://blog.csdn.net/qq_22532597/article/details/80314896

https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/8.0_20171129/cudnn-8.0-linux-x64-v7



### 尝试更改cudnn版本

现在把

/usr/local/cuda 里的cudnn换成了7.0.5

改了一下root/.bashrc



<https://blog.csdn.net/Lucifer_zzq/article/details/76675239>

<https://blog.csdn.net/haohaibo031113/article/details/71104088>

<https://blog.csdn.net/zl535320706/article/details/81979825>



```shell
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda
```

还是报错

```shell
2019-12-13 11:11:36.829100: E tensorflow/stream_executor/cuda/cuda_dnn.cc:378] Loaded runtime CuDNN library: 7103 (compatibility version 7100) but source was compiled with 7005 (compatibility version 7000).  If using a binary install, upgrade your CuDNN library to match.  If building from sources, make sure the library loaded at runtime matches a compatible version specified during compile configuration.
2019-12-13 11:11:36.830459: F tensorflow/core/kernels/conv_ops.cc:667] Check failed: stream->parent()->GetConvolveAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms)
2019-12-13 11:11:38.672160: E tensorflow/stream_executor/cuda/cuda_dnn.cc:378] Loaded runtime CuDNN library: 7103 (compatibility version 7100) but source was compiled with 7005 (compatibility version 7000).  If using a binary install, upgrade your CuDNN library to match.  If building from sources, make sure the library loaded at runtime matches a compatible version specified during compile configuration.
2019-12-13 11:11:38.675220: F tensorflow/core/kernels/conv_ops.cc:667] Check failed: stream->parent()->GetConvolveAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms)

```







# failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED

```shell
2019-12-11 19:55:08.834744: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
2019-12-11 19:55:10.232859: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: Tesla P40 major: 6 minor: 1 memoryClockRate(GHz): 1.531
pciBusID: 0000:3b:00.0
totalMemory: 22.38GiB freeMemory: 21.20GiB
2019-12-11 19:55:11.275901: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 1 with properties: 
name: Tesla P40 major: 6 minor: 1 memoryClockRate(GHz): 1.531
pciBusID: 0000:86:00.0
totalMemory: 22.38GiB freeMemory: 22.21GiB
2019-12-11 19:55:12.314189: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 2 with properties: 
name: Tesla P40 major: 6 minor: 1 memoryClockRate(GHz): 1.531
pciBusID: 0000:af:00.0
totalMemory: 22.38GiB freeMemory: 22.21GiB
2019-12-11 19:55:12.317230: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Device peer to peer matrix
2019-12-11 19:55:12.317543: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1051] DMA: 0 1 2 
2019-12-11 19:55:12.317801: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 0:   Y N N 
2019-12-11 19:55:12.317845: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 1:   N Y N 
2019-12-11 19:55:12.317956: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1061] 2:   N N Y 
2019-12-11 19:55:12.318023: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: Tesla P40, pci bus id: 0000:3b:00.0, compute capability: 6.1)
2019-12-11 19:55:12.318160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:1) -> (device: 1, name: Tesla P40, pci bus id: 0000:86:00.0, compute capability: 6.1)
2019-12-11 19:55:12.318284: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:2) -> (device: 2, name: Tesla P40, pci bus id: 0000:af:00.0, compute capability: 6.1)
2019-12-11 19:55:14.332396: E tensorflow/stream_executor/cuda/cuda_blas.cc:366] failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED
2019-12-11 19:55:16.042176: E tensorflow/stream_executor/cuda/cuda_blas.cc:366] failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED
2019-12-11 19:55:16.042435: W tensorflow/stream_executor/stream.cc:1901] attempting to perform BLAS operation using StreamExecutor without BLAS support
Traceback (most recent call last):
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1323, in _do_call
    return fn(*args)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1302, in _run_fn
    status, run_metadata)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py", line 473, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.InternalError: Blas GEMM launch failed : a.shape=(10, 10), b.shape=(10, 10), m=10, n=10, k=10
	 [[Node: MatMul = MatMul[T=DT_FLOAT, transpose_a=false, transpose_b=false, _device="/job:localhost/replica:0/task:0/device:GPU:0"](random_uniform, transpose)]]
	 [[Node: Sum/_1 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_19_Sum", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "test_GPU_1.py", line 39, in <module>
    result = session.run(sum_operation)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 889, in run
    run_metadata_ptr)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1120, in _run
    feed_dict_tensor, options, run_metadata)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1317, in _do_run
    options, run_metadata)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/client/session.py", line 1336, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InternalError: Blas GEMM launch failed : a.shape=(10, 10), b.shape=(10, 10), m=10, n=10, k=10
	 [[Node: MatMul = MatMul[T=DT_FLOAT, transpose_a=false, transpose_b=false, _device="/job:localhost/replica:0/task:0/device:GPU:0"](random_uniform, transpose)]]
	 [[Node: Sum/_1 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_19_Sum", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

Caused by op 'MatMul', defined at:
  File "test_GPU_1.py", line 14, in <module>
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py", line 1891, in matmul
    a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/ops/gen_math_ops.py", line 2437, in _mat_mul
    name=name)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 2956, in create_op
    op_def=op_def)
  File "/home/lqz/anaconda3/envs/withtensorflow15/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1470, in __init__
    self._traceback = self._graph._extract_stack()  # pylint: disable=protected-access

InternalError (see above for traceback): Blas GEMM launch failed : a.shape=(10, 10), b.shape=(10, 10), m=10, n=10, k=10
	 [[Node: MatMul = MatMul[T=DT_FLOAT, transpose_a=false, transpose_b=false, _device="/job:localhost/replica:0/task:0/device:GPU:0"](random_uniform, transpose)]]
	 [[Node: Sum/_1 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_19_Sum", tensor_type=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]

```

经过实验，加法不会出错，乘法会出错 

解决方法

使用root 用户运行

测试代码如下

https://blog.csdn.net/beyond9305/article/details/90450246



# TODO 我想跑代码 呜呜呜

## 1

gpu02

withtensorflow15

用户 lqz


```
E tensorflow/stream_executor/cuda/cuda_blas.cc:366] failed to create cublas handle: CUBLAS_STATUS_NOT_INITIALIZED

E tensorflow/stream_executor/cuda/cuda_dnn.cc:385] could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
```


## 2

gpu03

withtensorflow15

用户lqz

```shell
2019-12-11 21:09:05.072618: E tensorflow/stream_executor/cuda/cuda_dnn.cc:378] Loaded runtime CuDNN library: 7103 (compatibility version 7100) but source was compiled with 7005 (compatibility version 7000).  If using a binary install, upgrade your CuDNN library to match.  If building from sources, make sure the library loaded at runtime matches a compatible version specified during compile configuration.
2019-12-11 21:09:05.074101: F tensorflow/core/kernels/conv_ops.cc:667] Check failed: stream->parent()->GetConvolveAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms) 
2
```



## 3

gpu02

withtensorflow15

用户root

```shell
2019-12-11 20:05:47.672003: E tensorflow/stream_executor/cuda/cuda_dnn.cc:378] Loaded runtime CuDNN library: 7103 (compatibility version 7100) but source was compiled with 7005 (compatibility version 7000).  If using a binary install, upgrade your CuDNN library to match.  If building from sources, make sure the library loaded at runtime matches a compatible version specified during compile configuration.
2019-12-11 20:05:47.676984: F tensorflow/core/kernels/conv_ops.cc:667] Check failed: stream->parent()->GetConvolveAlgorithms( conv_parameters.ShouldIncludeWinogradNonfusedAlgo<T>(), &algorithms) 
```

死马当活马医

https://blog.csdn.net/qq_33547191/article/details/82596770

真的可以，但不用GPU 要慢一点