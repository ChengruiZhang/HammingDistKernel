
# demo验证

本算子仅作为demo验证，在8.3.RC1环境下运行

```
  cd kernel_launch_method_by_direct
  bash run.sh # 编译
  bash test.sh # 运行
```

在batch=1，seqLen=512，headQ=1，headK=1，hdim=128，topk=4，buffer=2，device_id=5的配置下

整体算子延迟为155us

```
  Op Name,Op Type,Task Duration(us),Block Dim,Mix Block Dim,Device Id,Pid,Current Freq,Rated Freq,
  topk_custom_0,vector,155.740005,1,NA,4,1304945,1800,1800,
```

去除掉汉明距离计算后的延迟为23.16us

```
  Op Name,Op Type,Task Duration(us),Block Dim,Mix Block Dim,Device Id,Pid,Current Freq,Rated Freq,
  topk_custom_0,vector,23.160000,1,NA,5,1252295,800,1800,
```

说明中间计算过程花费了130+us，远超预期，故验证该方法不可行

