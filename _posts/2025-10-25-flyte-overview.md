---
layout: post
title: flyteML编排-综述
date: 2025-10-25 10:35:00.000000000 +08:00
---

> Flyte 是一个面向机器学习、数据工程和分析工作流的云原生工作流编排平台。它由Lyft开发并开源，目前是Linux Foundation AI & Data下的一个孵化级项目。它的核心设计目标是让用户能够以 可复现、可扩展、类型安全 的方式定义、运行和管理复杂的数据/ML工作流。目前该项目已经可以进行生产级别的部署，并可支撑运行10000+级别的相关workflow任务。

## 1. 组件与架构

flyte有如下一些特点，在用户可以很方便的使用，并且插件集丰富，可以轻松扩展到数据处理等其他任务。

| 特性 | 解释 |
|-----|-----|
| K8S 原生运行 | 执行引擎基于K8S，所有任务都是pod运行，天然支持容器化和资源隔离，可大规模部署 |
| 易用的SDK | SDK(flytekit)通过简单的装饰函数@task、@workflow即可定义和串联起整个DAG，并可自定义ImageSpec、Container等镜像和运行时信息 |
| 自动缓存与重试 | Flyte会通过任务hash计算对输出进行cache，避免重复计算，失败时可重试 |
| 版本及可视化 | 所有WF都会版本化，并通过flyte console页面进行可视化 |
| 丰富的插件集 | Flyte在SDK和后端服务都设计了可配置的插件系统，可以轻松运行Kubeflow、KubeRay、Kserve、Airflow、Argo、Dash等自系统任务 |
| 覆盖MLOps全生命周期 | 通过以上特性的组合，flyte支持了数据预处理、实验、训练、模型注册、推理等完整的ML生命周期跟踪 |

flyte整个项目由以下几个组件组成:
| 组件 | 用途 |
|-----|-----|
| FlyteKit | Python SDK，定义任务、工作流、Launch Plan等，当然也包括pyflyte，flytectl等命令行工具 |
| FlyteConsole | Web UI，支持查看任务、执行日志、版本、数据流图等 |
| FlyteAdmin | 控制平面（Control Plane），管理任务、工作流定义、执行记录、权限和调度、提交CRD等 |
| FlytePropeller | 数据平面（Data Plane），负责在Kubernetes上调度和执行工作流 DAG |
| FlyteScheduler | 定时任务触发器，支持周期性执行工作流（可选择性开启）|
| FlyteSnacks | Flyte 用户使用示例，包含了基础操作、插件使用样例 |

### flyte整体架构图

flyte目前依赖postgres和minio两个公共服务用于存储数据，整体架构图如下
![flyte_overview](/images/flyte/1/flyte-overview.png)

### 分层定义

1. 用户层

    flyte与用户交互入口，通过命令行、WebUI、SDK等工具提供工作流、运行任务、工作空间、插件组件的交互操作

   * pyflyte：为在容器或者本地运行工作流的命令行工具, 面向开发者使用, 主要用途是运行workflow任务(local/remote)、构建镜像、pb构建、本地缓存管理、创建flyte模板项目、打包代码文件、创建运行计划等。
   * flytectl：为与后端系统控制交互的命令行工具, 面向flyte平台管理运维者使用，主要用途是查看运行计划、执行实例、任务 运行状态、查看任务定义等。
   * SDK：为用户提供灵活的task、workflow、imagespc、plugins的定义和组合，方便通过代码直接构建DAG，并与后端服务交互发起workflow。

2. 控制层(admin)

    为flyte的管理和调度中心，主要负责任务注册、调度、执行状态管理、版本控制与权限治理等

    * 。