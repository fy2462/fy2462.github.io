---
layout: post
title: 图解vllm-model与attention_backend
date: 2024-10-03 15:35:00.000000000 +08:00
---

> 通过上文可以了解到，执行器在调用推理接口后，最终将请求传入worker中的ModelRunner进行推理计算，这时如何使用CUDA加速模型attention和forward的推理。另外，对于我们定制化的model是如何加入到vllm的推理框架来使用的。通过本文你可以了解到这一过程背后的逻辑关系。

## 1. 模型推理图解

![atten_backend](/images/vllm/5/atten_backend.png)

* 从图解中可以看到，ModelRunner拥有attn_backend、graph_runner、att_state、model, 其中attn_backend在不同平台上使用的加速算法；graph_runner主要是在每个Paralletl Pipeline中的推理过程中构建cuda graph优化实例；att_state这个就是attention权重了; model就是我们所说的模型，我们使用其中的forward和sample用于MLP和Token sample策略的输出。
* 调用方法上看，主要分为load_model、capture_model、execute_models三类函数，load_model上篇文章已经讲过，我们针对其他2个方法进行讲解。

## 2. 模型cuda graph加速

* 上个章节中，capture_model这个方法调用是在initialize_cache(线条3)中进行的，最终在初始化完cache engine后，我们需要使用一个fake的数据对模型进行预热，并构建模型执行推理的时候的CUDA Graph用于加速。

![atten_backend](/images/vllm/5/graph_model.png)

* 我们看一下capture_model的主要流程, 这里主要是通过

```python
@torch.inference_mode()
def capture_model(self, kv_caches: List[List[torch.Tensor]]) -> None:
    """Cuda graph capture a model.
    请注意,如果批处理token的数量大于200,则CUDA图的性能增益可以忽略不计。
    而且由于CUDA图需要固定大小的张量,因此支持大/可变的批处理大小需要很高的GPU内存开销。
    因此,vLLM仅捕获 解码 请求。混合批处理（分块预填充 + 解码）或预填充请求不会被捕获。
    由于它仅用于解码,因此它假设批处理中每个序列只有1个token.
    """
    start_time = time.perf_counter()

    # Prepare dummy inputs. These will be reused for all batch sizes.
    max_batch_size = self.max_batchsize_to_capture
    # 虚拟推理数据
    input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
    input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
    if self.model_is_mrope:
        input_positions = torch.tile(input_positions, (3, 1))
    # Prepare dummy previous_hidden_states only if needed by the model.
    # This is used by draft models such as EAGLE.
    previous_hidden_states = None
    # 模型的实际调用方法中的参数中有previous_hidden_states
    if "previous_hidden_states" in inspect.signature(
            self.model.forward).parameters:
        previous_hidden_states = torch.empty(
            [max_batch_size,
                self.model_config.get_hidden_size()],
            dtype=self.model_config.dtype,
            device=self.device)

    intermediate_inputs = None
    if not get_pp_group().is_first_rank:
        intermediate_inputs = self.model.make_empty_intermediate_tensors(
            batch_size=max_batch_size,
            dtype=self.model_config.dtype,
            device=self.device)

    # Prepare buffer for outputs. These will be reused for all batch sizes.
    # It will be filled after the first graph capture.
    # 默认每一个隐藏状态为None
    hidden_or_intermediate_states: List[Optional[torch.Tensor]] = [
        None
    ] * self.parallel_config.pipeline_parallel_size

    graph_batch_size = self.max_batchsize_to_capture
    batch_size_capture_list = [
        bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
    ]

    # 创建graph stream，并返回关联的context
    with self.attn_state.graph_capture(
            max_batch_size), graph_capture() as graph_capture_context:
        # NOTE: Capturing the largest batch size first may help reduce the
        # memory usage of CUDA graph.
        # 对于每个pp数组执行同样的graph
        for virtual_engine in range(
                self.parallel_config.pipeline_parallel_size):
            # 所有满足graph size的batch size
            for batch_size in reversed(batch_size_capture_list):
                # FlashAttentionMetadata for CUDA
                attn_metadata = (
                    self.attn_state.graph_capture_get_metadata_for_batch(
                        batch_size,
                        is_encoder_decoder_model=self.model_config.
                        is_encoder_decoder_model))

                if self.lora_config:
                    lora_mapping = LoRAMapping(
                        **dict(index_mapping=[0] * batch_size,
                                prompt_mapping=[0] * batch_size,
                                is_prefill=False))
                    self.set_active_loras(set(), lora_mapping)

                if self.prompt_adapter_config:
                    prompt_adapter_mapping = PromptAdapterMapping(
                        [-1] * batch_size,
                        [-1] * batch_size,
                    )
                    self.set_active_prompt_adapters(
                        set(), prompt_adapter_mapping)
                # 先执行forward
                # this.graph_runner(...) -> graph_runner.forward(...)
                graph_runner = CUDAGraphRunner(
                    self.model, self.attn_backend.get_name(),
                    self.attn_state.graph_clone(batch_size),
                    self.model_config.is_encoder_decoder_model)

                capture_inputs = {
                    "input_ids":
                    input_tokens[:batch_size],
                    "positions":
                    input_positions[..., :batch_size],
                    "hidden_or_intermediate_states":
                    hidden_or_intermediate_states[
                        virtual_engine]  # type: ignore
                    [:batch_size]
                    if hidden_or_intermediate_states[virtual_engine]
                    is not None else None,
                    "intermediate_inputs":
                    intermediate_inputs[:batch_size]
                    if intermediate_inputs is not None else None,
                    "kv_caches":
                    kv_caches[virtual_engine],
                    "attn_metadata":
                    attn_metadata,
                    "memory_pool":
                    self.graph_memory_pool,
                    "stream":
                    graph_capture_context.stream
                }
                if previous_hidden_states is not None:
                    capture_inputs[
                        "previous_hidden_states"] = previous_hidden_states[:
                                                                            batch_size]

                if self.has_seqlen_agnostic:
                    # Only used by Mamba-based models CUDA graph atm (Jamba)
                    capture_inputs.update({
                        "seqlen_agnostic_capture_inputs":
                        self.model.get_seqlen_agnostic_capture_inputs(
                            batch_size)
                    })
                if self.model_config.is_encoder_decoder_model:
                    # add the additional inputs to capture for
                    # encoder-decoder models.
                    self._update_inputs_to_capture_for_enc_dec_model(
                        capture_inputs)
                # capture graph，获取 self._graph, 此时就缓存到runner中了
                graph_runner.capture(**capture_inputs)
                self.graph_memory_pool = graph_runner.graph.pool()
                self.graph_runners[virtual_engine][batch_size] = (
                    graph_runner)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # This usually takes < 10 seconds.
    logger.info("Graph capturing finished in %.0f secs.", elapsed_time)
```