import os
import types

from flash_attn import flash_attn_func

import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

from tts.modules.llm_dit.transformer import apply_rotary_emb


if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


#-------------------utils hooks-------------------
"""
Calculate FLOPs for attention operations
"""

def calculate_flops_hook(module, args, kwargs):
    hidden_states = args[0]
    batch_size, seq_len, dim = hidden_states.shape
    
    # Base operations: Q*K + Attention*V
    base_ops = seq_len * seq_len * module.n_local_heads * batch_size * dim // module.n_local_heads + seq_len * dim * batch_size * seq_len
    
    # Record full operations
    module.full_ops += base_ops
    
    # Get current method
    method = module.steps_method[module.step]
    
    # Calculate actual operations based on method
    if method == "BS":
        base_ops = base_ops / 3
    elif method == "TS":
        base_ops = 0
    
    # Record efficient operations
    module.efficient_ops += base_ops

def calculate_ff_flops_hook(module, args, kwargs):
    hidden_states = args[0]
    batch_size, seq_len, dim = hidden_states.shape
    
    inner_dim = module.w1.out_features
    
    # Base operations include:
    # 1. First Linear: dim -> inner_dim
    # 2. SiLU activation: element-wise on inner_dim
    # 3. Second Linear: inner_dim -> dim
    base_ops = (
        batch_size * seq_len * dim * inner_dim +  # First Linear 
        batch_size * seq_len * inner_dim +        # SiLU activation
        batch_size * seq_len * inner_dim * dim    # Second Linear
    )
    
    module.full_ops += base_ops
        
    # Calculate actual operations based on method
    method = module.steps_method[module.step]
    if method == "TS":
        base_ops = 0
    elif method == "BS":
        base_ops /= 3
    
    # Record efficient operations
    module.efficient_ops += base_ops

"""
Calculate compression loss between raw output and efficient output using default values
"""
def compression_loss(a, b, metrics = "l1"):
    """
    Calculate the compression loss between two sets of tensors.

    Args:
        a (list of torch.Tensor): First set of tensors, usually the raw output
        b (list of torch.Tensor): Second set of tensors to compare against, usually the compressed output

    Returns:
        l (float): The calculated loss value.
    """
    ls = []
    for ai, bi in zip(a, b):
        diff = (ai - bi) / (torch.max(ai, bi) + 1e-6)
        l = diff.abs().clip(0, 10).mean()
        ls.append(l)
    l = sum(ls) / len(ls)
    return l

def pre_calibration_hook(module, args, kwargs):
    """Pre-calibration: Determine greedy search patterns by comparing heatmaps with diagonal at each layer and timestep"""
    # Get current timestep
    step = module.step
    
    # Save weights
    x = args[0]
    start_pos = args[1] if len(args) > 1 else kwargs.get('start_pos', 0)
    
    bsz, seqlen, _ = x.shape
    
    # Calculate query and key
    xq = module.wq(x)
    xk = module.wk(x)
    # Use higher precision for matrix multiplication
    xq_64 = xq.to(torch.float64)
    xk_64 = xk.to(torch.float64)
    # Calculate attention scores
    scale = 1.0 / math.sqrt(module.head_dim)
    attn_scores = torch.matmul(xq_64, xk_64.transpose(-2, -1)) * scale
    attn_scores_stable = attn_scores - torch.max(attn_scores, dim=-1, keepdim=True)[0]
    attn_weights = F.softmax(attn_scores_stable, dim=-1)
    # Check for NaN after softmax
    if torch.isnan(attn_weights).any():
        print(f"Warning: NaN detected after softmax at step {step}")
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    # Create diagonal matrix
    diagonal_matrix = torch.eye(seqlen, device=attn_weights.device, dtype=attn_weights.dtype)
    
    # Calculate cosine similarity
    attn_weights_cond, attn_weights_uncond = attn_weights.chunk(2, dim=0)
    batch_size = attn_weights_cond.shape[0]
    similarities = []
    
    for b in range(batch_size):
        # Get attention weights for current batch, take first head
        attn_mat = attn_weights_cond[b]
        
        # Flatten matrix to vector
        attn_vec = attn_mat.reshape(-1)
        diag_vec = diagonal_matrix.reshape(-1)
        
        # Calculate cosine similarity
        attn_norm = attn_vec / (torch.norm(attn_vec) + 1e-6)
        diag_norm = diag_vec / (torch.norm(diag_vec) + 1e-6)
        
        sim = torch.dot(attn_norm, diag_norm)
        similarities.append(sim.item())
    
    # Average similarity
    similarity_ts = sum(similarities) / len(similarities)
    
    # Save similarity
    if not hasattr(module, 'diagonal_similarities'):
        module.diagonal_similarities = {}
    module.diagonal_similarities[step] = similarity_ts
    
    module.step += 1 # if step is not added in forward, add it here
    

def pre_calibration(model,steps=32, threshold=0.1):
    '''
    Pre-calibration Phase
    '''
    print("Pre Calibration for transformer!!!")
    transformer = model.dit.encoder # model should be megatts3infer/diffusion.encoder
    
    loss_thresholds = []
    for step_i in range(steps):
        sub_list = []
        for blocki in range(len(transformer.layers)):
            threshold_i = (blocki + 1) / len(transformer.layers) * threshold
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)

    calibration_preparation(transformer)

    hook = transformer.register_forward_pre_hook(transformer_forward_pre_hook_for_pre_calibration, with_kwargs=True)
    transformer.loss_thresholds = loss_thresholds
    return hook # Return hook reference for removal
    

def pre_calibration_check(model):
    '''
    Explore which blocks can prioritize TS/BS mechanisms
    '''
    print("Pre Calibration Exploring for transformer!!!")
    transformer = model.dit.encoder # model should be megatts3infer/diffusion.encoder
    # Disable cache
    calibration_reset(transformer) # to set step in module
    for block in transformer.layers:
        block.attention.need_cache_output = [False] * 32 # magic num, it should be nfe_steps
        block.feed_forward.need_cache_output = [False] * 32 # magic num, it should be nfe_steps
    hooks = []
    for blocki in range(len(transformer.layers)):
        block = transformer.layers[blocki]
        hooks.append(block.attention.register_forward_pre_hook(pre_calibration_hook, with_kwargs=True))
    return hooks

'''
Pre-calibration
'''
def transformer_forward_pre_hook_for_pre_calibration(model, args, kwargs):
    
    now_stepi = model.layers[0].attention.step # model should be encoder
    print(f"Pre Calibration Step: {now_stepi}")

    # Disable cache during method search to avoid modifications
    for block in model.layers:
        block.attention.forward = types.MethodType(cuda_timer(efficient_attention_forward), block.attention)
        block.attention.need_cache_output[now_stepi] = False
        block.feed_forward.need_cache_output[now_stepi] = False
    
    # First run to get full-attention values. In pre-calibration, focus on ts_first blocks and try setting them to TS
    raw_outputs = model.forward(*args, **kwargs)
    raw_output_cond_spktxt,raw_output_condtxt,raw_output_uncond = raw_outputs.chunk(3,dim=0)
    raw_outputs = raw_output_uncond + 2.5 * (raw_output_cond_spktxt - raw_output_uncond) + 1.6 * (raw_output_cond_spktxt - raw_output_condtxt)
    for blocki, block in enumerate(model.layers):
        if now_stepi == 0:
            continue
        attn = block.attention
        ff =block.feed_forward
        assert hasattr(attn, 'ts_first'), "attn.ts_first not found"
        if attn.ts_first[now_stepi] is False:
            continue
        elif attn.ts_first[now_stepi] is True:
            method_candidates = ['TS']
        
        selected_method = 'NONE'
        for method in method_candidates:
            attn.steps_method[now_stepi] = method
            ff.steps_method[now_stepi] = method

            for block_ in model.layers:
                block_.attention.step = now_stepi
                block_.feed_forward.step = now_stepi
            efficient_outputs = model.forward(*args, **kwargs)
            efficient_output_cond_spktxt,efficient_output_condtxt, efficient_output_uncond = efficient_outputs.chunk(3,dim=0)
            efficient_outputs = efficient_output_uncond + 2.5 * (efficient_output_cond_spktxt - efficient_output_uncond) + 1.6 * (efficient_output_cond_spktxt - efficient_output_condtxt)
            
            loss = compression_loss(raw_outputs, efficient_outputs)
            threshold = model.loss_thresholds[now_stepi][blocki]

            if loss<threshold:
                remaining = len(method_candidates) - method_candidates.index(method)
                selected_method = method
                break

        
        attn.steps_method[now_stepi] = selected_method
        ff.steps_method[now_stepi] = selected_method
        del loss, efficient_outputs

    del raw_outputs

    # As this is just a transformer prehook,
    # step will increment in the final forward pass after all mechanisms are determined
    # so we need to restore the incremented step here
    for block_ in model.layers:
        block_.attention.step = now_stepi
        block_.feed_forward.step = now_stepi

    # Enable cache for this step after plan is determined
    for block in model.layers:
        block.attention.need_cache_output[now_stepi] = True
        block.feed_forward.need_cache_output[now_stepi] = True


"""
Calibration function
"""
def transformer_forward_pre_hook_for_calibration(model, args, kwargs):
    
    now_stepi = model.layers[0].attention.step
    print(f"Calibration Step: {now_stepi}")

    # Disable cache during method search to avoid modifications
    for block in model.layers:
        block.attention.forward = types.MethodType(cuda_timer(efficient_attention_forward), block.attention)
        block.attention.need_cache_output[now_stepi] = False
        block.feed_forward.need_cache_output[now_stepi] = False

    # Total progress bar
    total_blocks = len(model.layers)
    
    # Current step progress bar
    step_pbar = tqdm(
        total=total_blocks * 2,
        desc=f"Step {now_stepi}/32",
        position=1,
        leave=False,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]',
        postfix=f"block 0/{total_blocks} method: initializing"
    )

    # First run to get full-attention values
    raw_outputs = model.forward(*args, **kwargs)
    raw_output_cond_spktxt,raw_output_condtxt,raw_output_uncond = raw_outputs.chunk(3,dim=0)
    raw_outputs = raw_output_uncond + 2.5 * (raw_output_cond_spktxt - raw_output_uncond) + 1.6 * (raw_output_cond_spktxt - raw_output_condtxt)
    
    nots = model.nots
    nobs = model.nobs
    
    for blocki, block in enumerate(model.layers):
        if now_stepi == 0:
            continue
        attn = block.attention
        assert hasattr(attn, 'ts_first') , "attn.ts_first not found"
        if attn.steps_method[now_stepi] == 'TS': # Pre-calibration already checked, skip directly
            step_pbar.update(2)
            continue
        if not nots and not nobs:
            method_candidates = ['TS', 'BS']
        elif nots:
            method_candidates = ['BS']
        elif nobs:
            method_candidates = ['TS']
        selected_method = 'NONE'
        for method in method_candidates:
            step_pbar.set_postfix_str(f"block {blocki + 1}/{total_blocks} method: {method}")
            block.attention.steps_method[now_stepi] = method
            block.feed_forward.steps_method[now_stepi] = method

            for block_ in model.layers:
                block_.attention.step = now_stepi
                block_.feed_forward.step = now_stepi
            efficient_outputs = model.forward(*args, **kwargs)
            efficient_output_cond_spktxt,efficient_output_condtxt, efficient_output_uncond = efficient_outputs.chunk(3,dim=0)
            efficient_outputs = efficient_output_uncond + 2.5 * (efficient_output_cond_spktxt - efficient_output_uncond) + 1.6 * (efficient_output_cond_spktxt - efficient_output_condtxt)
            
            loss = compression_loss(raw_outputs, efficient_outputs)
            threshold = model.loss_thresholds[now_stepi][blocki]

            if loss<threshold:
                remaining = len(method_candidates) - method_candidates.index(method)
                step_pbar.update(remaining)
                selected_method = method
                break
            
            step_pbar.update(1)
            
        step_pbar.close()
        
        block.attention.steps_method[now_stepi] = selected_method
        block.feed_forward.steps_method[now_stepi] = selected_method
        del loss, efficient_outputs
        
    del raw_outputs

    # As this is just a transformer prehook,
    # step will increment in the final forward pass after all mechanisms are determined
    # so we need to restore the incremented step here
    for block_ in model.layers:
        block_.attention.step = now_stepi
        block_.feed_forward.step = now_stepi

    # Enable cache for this step after plan is determined
    for block in model.layers:
        block.attention.need_cache_output[now_stepi] = True
        block.feed_forward.need_cache_output[now_stepi] = True

def calibration(model, steps=32, threshold=0.1):

    print("Calibration for transformer!!!")
    transformer = model.dit.encoder # model should be megatts3infer/diffusion.encoder

    loss_thresholds = []
    for step_i in range(steps):
        sub_list = []
        for blocki in range(len(transformer.layers)):
            threshold_i = (blocki + 1) / len(transformer.layers) * threshold
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)

    calibration_preparation(transformer, is_method_init=False)

    hook = transformer.register_forward_pre_hook(transformer_forward_pre_hook_for_calibration, with_kwargs=True)
    transformer.loss_thresholds = loss_thresholds
    return hook # Return hook reference for removal

def speedup(model,delta = None, steps=32):
    assert delta is not None, "delta should be set"
    print("Speedup for transformer!!!")
    transformer = model.dit.encoder # model should be megatts3infer/diffusion.encoder
    # Load methods
    path = f"data/methods/{steps}_{delta}.json"
    calibration_preparation(transformer, steps=steps, method_path = path)


def calibration_preparation(transformer, steps=32, method_path = None,is_method_init=True):

    if method_path is None:
        for i, block in enumerate(transformer.layers):
            attn = block.attention
            ff = block.feed_forward
            # Set attributes for attention
            attn.id = i
            attn.step = 0
            attn.total_latency = 0.0
            attn.full_ops = 0
            attn.efficient_ops = 0
            attn.forward = types.MethodType(cuda_timer(efficient_attention_forward), attn)
            if is_method_init:
                attn.steps_method = ['NONE'] * steps
            attn.need_cache_output = [True] * steps
            attn.cached_output = None
            # Set attributes for feed-forward
            if is_method_init:
                ff.steps_method = ['NONE'] * steps
            ff.need_cache_output = [True] * steps
            ff.full_ops = 0
            ff.efficient_ops = 0
            ff.step = 0
            ff.total_latency = 0.0
            ff.forward = types.MethodType(cuda_timer(efficient_ff_forward), ff)
            ff.cached_output = None
            ff.id = i
    else:
        with open(method_path, 'r') as f:
            import json
            saved_methods = json.loads(open(method_path).read())['methods']

            for i, (methods, block) in enumerate(zip(saved_methods, transformer.layers)):
                # Set attributes for attention
                attn = block.attention
                attn.steps_method = methods
                attn.id = i
                attn.step = 0
                attn.total_latency = 0.0
                attn.full_ops = 0
                attn.efficient_ops = 0
                attn.forward = types.MethodType(cuda_timer(efficient_attention_forward), attn)
                attn.need_cache_output = [True] * steps
                attn.cached_output = None
                # Set attributes for feed-forward
                ff = block.feed_forward
                ff.steps_method = methods
                ff.id = i
                ff.need_cache_output = [True] * steps
                ff.step = 0
                ff.total_latency = 0.0
                ff.full_ops = 0
                ff.efficient_ops = 0
                ff.forward = types.MethodType(cuda_timer(efficient_ff_forward), ff)
                ff.cached_output = None

# After every calibration phase, module steps should be reset
def calibration_reset(transformer, steps=32):
    for block in transformer.layers:
        attn = block.attention
        ff = block.feed_forward
        # Reset attributes for attention
        attn.step = 0
        attn.total_latency = 0.0
        attn.need_cache_output = [True] * steps
        attn.cached_output = None
        # Reset attributes for feed-forward
        ff.need_cache_output = [True] * steps
        ff.step = 0
        ff.total_latency = 0.0
        ff.cached_output = None

def eval_reset(transformer, steps=32):
    for block in transformer.layers:
        attn = block.attention
        ff = block.feed_forward
        # Reset attributes for attention
        attn.step = 0
        attn.need_cache_output = [True] * steps
        attn.cached_output = None
        # Reset attributes for feed-forward
        ff.need_cache_output = [True] * steps
        ff.step = 0
        ff.cached_output = None

def efficient_ff_forward(self, x):
    method = self.steps_method[self.step]
    if 'TS' in method:
        self.step += 1
        return self.cached_output
    elif 'BS' in method: # May need to keep first two branches
        x,_,_ = x.chunk(3,dim=0) # Batch size might not be 3, check if error occurs
        batch_size, _ , _ = x.shape
        out_cond = self.w2(F.silu(self.w1(x)))
        last_spktxt, last_txt, last_uncond = self.cached_output.chunk(3, dim = 0)
        uncond_res = last_uncond - last_spktxt
        txt_res = last_txt - last_spktxt
        out = torch.cat([out_cond, out_cond + txt_res, out_cond + uncond_res], dim=0)
        if self.need_cache_output[self.step]:
            self.cached_output = out
        self.step += 1
        return out
    elif 'NONE' in method:
        out = self.w2(F.silu(self.w1(x)))
        if self.need_cache_output[self.step]:
            self.cached_output = out
        self.step += 1
        return out
    else:
        raise NotImplementedError

def efficient_attention_forward(
    self,
    x: float, 
    start_pos: int, 
    freqs_cis,
    mask
): 
    
    method = self.steps_method[self.step]
    batch_size, seq_len, _ = x.shape
    # Directly share output from last step, TS mechanism
    if 'TS' in method:
        self.step += 1
        return self.cached_output

    # BS mechanism calculation
    # If using BS mechanism, calculate only conditional cases first    
    if 'BS' in method:
        # Exclude unconditional
        batch_size = batch_size // 3
        x = x[:batch_size]

    # `sample` projections.
    query = self.wq(x)
    key = self.wk(x)
    value = self.wv(x)

    batch_size, seq_len, _ = x.shape
    
    # attention
    query = query.view(batch_size, -1, self.n_local_heads, self.head_dim)
    key = key.view(batch_size, -1, self.n_local_kv_heads, self.head_dim)
    value = value.view(batch_size, -1, self.n_local_kv_heads, self.head_dim)
    query, key = apply_rotary_emb(query, key, freqs_cis=freqs_cis)

    output = flash_attn_func(query, key, value, causal=False)

    output = output.view(batch_size, seq_len, -1)
    x = output.to(query.dtype)
        
    # linear proj
    x = self.wo(x)
    
    if 'BS' in method:
        
        # Use existing cache to calculate residuals
        last_spktxt = self.cached_output[:batch_size]
        last_txt = self.cached_output[batch_size:2*batch_size]
        last_uncond = self.cached_output[2*batch_size:]
        if self.need_cache_output[self.step]:
            self.cached_output[:batch_size] = x
            self.cached_output[batch_size:2*batch_size] = x + last_txt - last_spktxt
            self.cached_output[2*batch_size:] = x + last_uncond - last_spktxt
            x = self.cached_output
        else:
            x = torch.cat([x, x + last_txt - last_spktxt, x + last_uncond - last_spktxt], dim=0)
    
    if 'NONE' in method and self.need_cache_output[self.step]:
        self.cached_output = x
    
    self.step += 1
        
    return x

def cuda_timer(func):
    '''
    A decorator to measure the latency of a function
    '''
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'total_latency'):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
        result = func(self, *args, **kwargs)
            
        if hasattr(self, 'total_latency'):
            end_event.record()
            torch.cuda.synchronize()
            self.total_latency += start_event.elapsed_time(end_event) / 1000.0  # latency in seconds
            
        return result
    return wrapper