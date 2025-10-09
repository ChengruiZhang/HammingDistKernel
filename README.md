
# demo验证

本算子仅作为demo验证，不可完整跑全shape下的情况。

核心代码在./kernel_impl/topk_custom.h

在8.3.RC1环境下运行

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

# 性能低效分析

1. 从HBM搬运数据到UB上的时候，每次搬运的数据量为[..., hidDim]，其中hidDim设置为128个bit，8个int16，但由于后续处理中，每次处理的数据必须是以16个int16为单位，且不同行的hidDim在后续需要做copy操作，无法直接填充在后面。这导致了每一次搬运的时候都需要对所有的hidDim进行padding，无法连续从内存搬运 -- 性能低下
2. 由于padding缘故，Vector算力也会有一半浪费
3. hamming距离的计算中有过多的vector计算+barrier（16*SeqLen个Vector计算），且右移、与等位操作并没有很高效，仍需要花费较多时间
