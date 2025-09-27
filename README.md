<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->

## 概述

本样例介绍了调用TopK高阶API实现topk单算子，并按照不同的算子调用方式分别给出了对应的端到端实现。

- 直调：使用核函数直调topk自定义算子。

  核函数的基础调用方式，开发者完成算子核函数的开发和Tiling实现后，即可通过AscendCL运行时接口，完成算子的调用。

- 框架调用：使用框架调用topk自定义算子。

  按照工程创建->算子实现->编译部署->算子调用的流程完成算子开发。整个过程都依赖于算子工程：基于工程代码框架完成算子核函数的开发和Tiling实现，通过工程编译脚本完成算子的编译部署，继而实现单算子调用或第三方框架中的算子调用。

本样例中包含如下调用方式：

| 调用方式  | 目录                                                         | **描述**                                                   |
| --------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| 直调    | [kernel_launch_method_by_direct](./kernel_launch_method_by_direct) | host侧的核函数调用程序，包含CPU侧、NPU侧、仿真侧三种运行验证方法。 |
| 框架调用 | [kernel_launch_method_by_framework](./kernel_launch_method_by_framework) | 通过aclnn调用的方式调用topk算子。                       |

## 样例支持的产品型号为：
- Atlas A2训练系列产品/Atlas 800I A2推理产品
- Atlas 推理系列产品AI Core

## 目录结构

| 目录                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [kernel_launch_method_by_direct](./kernel_launch_method_by_direct) | 通过kernel直调的方式调用自定义算子工程样例目录               |
| [kernel_launch_method_by_framework](./kernel_launch_method_by_framework) | 通过aclnn调用的方式调用自定义算子工程样例目录                |
| [host_tiling](./host_tiling)                                 | 本样例tiling代码实现 |
| [kernel_impl](./kernel_impl)                                 | 本样例kernel侧代码实现                                       |

## 算子描述

topk单算子，对输入tensor做TopK计算，用于获取最后一个维度的前k个最大值或最小值及其对应的索引。

topk算子规格：
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">TopKCustom</td></tr>

<tr><td rowspan="5" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">srcLocalValue</td><td align="center">2 * 32</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">srcLocalIndex</td><td align="center">32</td><td align="center">int32_t</td><td align="center">ND</td></tr>
<tr><td align="center">srcLocalFinish</td><td align="center">32</td><td align="center">int32_t</td><td align="center">ND</td></tr>


<tr><td rowspan="3" align="center">算子输出</td></tr>
<tr><td align="center">dstLocalValue</td><td align="center">2 * 8</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">dstLocalIndex</td><td align="center">2 * 8</td><td align="center">int32_t</td><td align="center">ND</td></tr>


<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">topk_custom</td></tr>
</table>

## 算子实现介绍

本样例中实现的是固定shape为输入srcLocalValue[2 * 32], srcLocalIndex[32], srcLocalFinish[32], 输出dstLocalValue[2 * 8], dstLocalIndex[2 * 8]的topk算子。

- kernel实现

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用TopK高阶API接口完成topk计算，得到最终结果，再搬出到外部存储上。

  topk算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGm存储在srcLocal中，Compute任务负责对srcLocal执行topk计算，计算结果存储在dstLocal中，CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGm。

- tiling实现

  topk算子的tiling实现流程如下：首先获取topk接口能完成计算所需最大/最小临时空间大小，根据该范围结合实际的内存使用情况设置合适的空间大小，然后根据输入长度dataLength确定所需tiling参数。