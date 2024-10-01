---
layout: post
title: 图解vllm-执行器与worker
date: 2024-10-01 16:34:00.000000000 +08:00
---

> 执行器（Executor）是对model worker的一层封装，LLMEngine会根据engine_config来创建确定创建哪个Executor，本文将以RayGPUExecutor为例进行介绍，Ray作为较为常用模型分布式框架，应用场景比较有代表性。

## 1. Ray模型执行器图解

![ray_executor](/images/vllm/4/ray_executor.png)

