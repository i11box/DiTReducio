import os
import types

from flash_attn import flash_attn_func

import torch
import torch.nn.functional as F
from tqdm import tqdm
from x_transformers.x_transformers import apply_rotary_pos_emb
import math

if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


#-------------------utils hooks-------------------
def calculate_flops_hook(module, args, kwargs):
    """
    Calculate FLOPs for attention operations.
    
    Args:
        module: The attention module
        args: Function arguments
        kwargs: Keyword arguments containing input tensor 'x'
        
    Notes:
        - Calculates base operations for attention: Q*K + Attention*V
        - Adjusts FLOPs based on optimization method:
            - BS: Halves base operations
            - TS: Zero operations
    """
    hidden_states = kwargs['x']
    batch_size, seq_len, dim = hidden_states.shape
    
    # base_op：Q*K + Attention*V
    base_ops = seq_len * seq_len * module.heads * batch_size * dim // module.heads + seq_len * dim * batch_size * seq_len
    
    module.full_ops += base_ops
    
    method = module.steps_method[module.step]
    
    # efficient_op
    if method == "BS":
        base_ops = base_ops / 2
    elif method == "TS":
        base_ops = 0
    
    module.efficient_ops += base_ops

def calculate_ff_flops_hook(module, args, kwargs):
    """
    Calculate FLOPs for feed-forward network operations.
    
    Args:
        module: The feed-forward module
        args: Function arguments containing input tensor
        kwargs: Keyword arguments
        
    Notes:
        - Calculates operations for feed-forward projections
        - Adjusts FLOPs based on optimization method:
            - TS: Zero operations 
            - BS: Half operations
    """
    hidden_states = args[0]
    batch_size, seq_len, dim = hidden_states.shape
    project_in = module.ff[0]
    first_linear = project_in[0]  
    inner_dim = first_linear.out_features

    # Calculate base operations
    base_ops = (
        batch_size * seq_len * dim * inner_dim +  # First Linear
        batch_size * seq_len * inner_dim * dim    # Second Linear
    )
    
    # Record full precision computation
    module.full_ops += base_ops
    
    # Get current method
    method = module.steps_method[module.step]
    
    # Calculate actual computation based on different methods
    if method == "TS":
        base_ops = 0
    elif method == "BS":
        base_ops *= 0.5
    
    # Record actual computation
    module.efficient_ops += base_ops

"""
Calculate the difference between raw output and efficient output, using default values to compute Loss
"""
def compression_loss(a, b):
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
    """Pre-calibration: Compare model heatmaps with diagonal at each layer and timestep to determine greedy search mode"""
    # Get current timestep
    step = module.step
    
    # Save weights
    x = kwargs['x']
    mask = kwargs.get('mask', None)
    
    query = module.to_q(x).to(dtype = torch.bfloat16) # flashattn inference required
    key = module.to_k(x).to(dtype = torch.bfloat16)
    
    inner_dim = key.shape[-1]
    attn_weights = query @ key.transpose(-2,-1) / math.sqrt(inner_dim) # Get attention weights
    if mask is not None:
        attn_weights = attn_weights.masked_fill(~mask, 0)
    attn_weights = F.softmax(attn_weights, dim=-1)
    _,n,_ = attn_weights.shape
    diagonal_matrix = torch.eye(n, device=attn_weights.device,dtype=attn_weights.dtype)
    
    # Calculate cosine similarity
    attn_weights_cond,attn_weights_uncond = attn_weights.chunk(2,dim = 0)
    batch_size = attn_weights_cond.shape[0]
    similarities = []

    for b in range(batch_size):
        # Get attention weights for current batch
        attn_mat = attn_weights_cond[b]
        
        # Flatten matrix to vector
        attn_vec = attn_mat.reshape(-1)
        diag_vec = diagonal_matrix.reshape(-1)
        
        # Calculate cosine similarity
        # Normalize vectors
        attn_norm = attn_vec / torch.norm(attn_vec)
        diag_norm = diag_vec / torch.norm(diag_vec)
        
        # Compute dot product
        sim = torch.dot(attn_norm, diag_norm) # Similarity between conditional branch and diagonal
        similarities.append(sim.item())
        
    # Take average as final similarity
    similarity_ts = sum(similarities) / len(similarities)
    
    # Record similarity
    if not hasattr(module, 'diagonal_similarities'):
        module.diagonal_similarities = {}
    module.diagonal_similarities[step] = similarity_ts        

def pre_calibration(model,steps=32, threshold=0.1):
    '''
    This is the formal pre-calibration
    '''
    print("Pre Calibration for transformer!!!")
    transformer = model.transformer # model should be cfm
    
    loss_thresholds = []
    for step_i in range(steps):
        sub_list = []
        for blocki in range(len(transformer.transformer_blocks)):
            threshold_i = (blocki + 1) / len(transformer.transformer_blocks) * threshold
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)

    calibration_preparation(transformer)

    hook = transformer.register_forward_pre_hook(transformer_forward_pre_hook_for_pre_calibration, with_kwargs=True)
    transformer.loss_thresholds = loss_thresholds
    return hook # Return hook reference for removal
    

def pre_calibration_check(model):
    '''
    This function is mainly used to explore which blocks can prioritize using ts/bs
    '''
    print("Pre Calibration Exploring for transformer!!!")
    transformer = model.transformer # model should be cfm
    # Turn off cache
    calibration_preparation(transformer)
    for block in transformer.transformer_blocks:
        block.attn.need_cache_output = [False] * 32 # magic num, it should be nfe_steps
        block.ff.need_cache_output = [False] * 32 # magic num, it should be nfe_steps
    hooks = []
    for blocki in range(len(transformer.transformer_blocks)):
        block = transformer.transformer_blocks[blocki]
        hooks.append(block.attn.register_forward_pre_hook(pre_calibration_hook, with_kwargs=True))
    return hooks

'''
Pre-calibration
'''
def transformer_forward_pre_hook_for_pre_calibration(model, args, kwargs):
    
    now_stepi = model.transformer_blocks[0].attn.step
    print(f"Pre Calibration Step: {now_stepi}")

    # To avoid changing cache content when searching for candidate methods, turn off cache switches during search
    for block in model.transformer_blocks:
        block.attn.forward = types.MethodType(cuda_timer(efficient_attention_forward), block.attn)
        block.attn.need_cache_output[now_stepi] = False
        block.ff.need_cache_output[now_stepi] = False
    
    # Run once to get full-attention values, in pre-calibration only focus on ts_first blocks and try setting them to ts
    raw_outputs = model.forward(*args, **kwargs)
    for blocki, block in enumerate(model.transformer_blocks):
        if now_stepi == 0:
            continue
        # Methods from strong to weak
        attn = block.attn
        assert hasattr(attn, 'ts_first'), "attn.ts_first not found"
        if attn.ts_first[now_stepi] is False:
            continue
        elif attn.ts_first[now_stepi] is True:
            method_candidates = ['TS']
        
        selected_method = 'NONE'
        for method in method_candidates:
            block.attn.steps_method[now_stepi] = method
            block.ff.steps_method[now_stepi] = method

            for block_ in model.transformer_blocks:
                block_.attn.step = now_stepi
                block_.ff.step = now_stepi
            efficient_outputs = model.forward(*args, **kwargs)
            loss = compression_loss(raw_outputs, efficient_outputs)
            threshold = model.loss_thresholds[now_stepi][blocki]

            if loss<threshold:
                remaining = len(method_candidates) - method_candidates.index(method)
                selected_method = method
                break

        block.attn.steps_method[now_stepi] = selected_method
        block.ff.steps_method[now_stepi] = selected_method
        del loss, efficient_outputs

    del raw_outputs

    # Since this is just a prehook for the transformer, after finalizing all mechanisms another transformer forward will run where step increments, so restore the incremented step here
    for block_ in model.transformer_blocks:
        block_.attn.step = now_stepi
        block_.ff.step = now_stepi

    # After determining the plan for this step, turn on cache switches to allow normal cache generation for this step
    for block in model.transformer_blocks:
        block.attn.need_cache_output[now_stepi] = True
        block.ff.need_cache_output[now_stepi] = True


"""
Calibration function: Use greedy search to calculate hyperparameters corresponding to each mechanism
"""
def transformer_forward_pre_hook_for_calibration(model, args, kwargs):
    
    now_stepi = model.transformer_blocks[0].attn.step
    print(f"Calibration Step: {now_stepi}")

    # To avoid changing cache content when searching for candidate methods, turn off cache switches during search
    for block in model.transformer_blocks:
        block.attn.forward = types.MethodType(cuda_timer(efficient_attention_forward), block.attn)
        block.attn.need_cache_output[now_stepi] = False
        block.ff.need_cache_output[now_stepi] = False

    # Total progress bar
    total_blocks = len(model.transformer_blocks)
    
    # Current step progress bar
    step_pbar = tqdm(
        total=total_blocks * 2,
        desc=f"Timestep {now_stepi}/32",
        position=1,
        leave=False,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{postfix}]',
        postfix=f"block 0/{total_blocks} method: initializing"
    )

    # Run once to get full-attention values
    raw_outputs = model.forward(*args, **kwargs)
    for blocki, block in enumerate(model.transformer_blocks):
        if now_stepi == 0:
            continue
        # Methods from strong to weak
        attn = block.attn
        assert hasattr(attn, 'ts_first') , "attn.ts_first not found"
        if attn.steps_method[now_stepi] == 'TS': # Pre-calibration already judged, so skip directly
            step_pbar.update(2)
            continue
        method_candidates = ['TS', 'BS']
        if model.nobs:
            method_candidates = ['TS']
        if model.nots:
            method_candidates = ['BS']
        selected_method = 'NONE'
        for method in method_candidates:
            step_pbar.set_postfix_str(f"block {blocki + 1}/{total_blocks} method: {method}")
            block.attn.steps_method[now_stepi] = method
            block.ff.steps_method[now_stepi] = method

            for block_ in model.transformer_blocks:
                block_.attn.step = now_stepi
                block_.ff.step = now_stepi
            efficient_outputs = model.forward(*args, **kwargs)
            loss = compression_loss(raw_outputs, efficient_outputs)
            threshold = model.loss_thresholds[now_stepi][blocki]

            if loss<threshold:
                remaining = len(method_candidates) - method_candidates.index(method)
                step_pbar.update(remaining)
                selected_method = method
                break
            
            step_pbar.update(1)
            
        step_pbar.close()
        
        block.attn.steps_method[now_stepi] = selected_method
        block.ff.steps_method[now_stepi] = selected_method
        del loss, efficient_outputs

    del raw_outputs

    # Since this is just a prehook for the transformer, after finalizing all mechanisms another transformer forward will run where step increments, so restore the incremented step here
    for block_ in model.transformer_blocks:
        block_.attn.step = now_stepi
        block_.ff.step = now_stepi

    # After determining the plan for this step, turn on cache switches to allow normal cache generation for this step
    for block in model.transformer_blocks:
        block.attn.need_cache_output[now_stepi] = True
        block.ff.need_cache_output[now_stepi] = True

def calibration(model, steps=32, threshold=0.1):#w

    print("Calibration for transformer!!!")
    transformer = model.transformer # model should be cfm

    loss_thresholds = []
    for step_i in range(steps):
        sub_list = []
        for blocki in range(len(transformer.transformer_blocks)):
            threshold_i = (blocki + 1) / len(transformer.transformer_blocks) * threshold
            sub_list.append(threshold_i)
        loss_thresholds.append(sub_list)

    calibration_preparation(transformer, is_method_init=False)

    hook = transformer.register_forward_pre_hook(transformer_forward_pre_hook_for_calibration, with_kwargs=True)
    transformer.loss_thresholds = loss_thresholds
    return hook # Return hook reference for removal

def speedup(model,delta = None, steps=32):#w
    assert delta is not None, "delta should be set"
    transformer = model.transformer 
    # Load methods
    path = "" # your path, e.g. "data/data/methods/{steps}_{delta}.json"
    calibration_preparation(transformer, steps=steps, method_path = path)

def calibration_preparation(transformer, steps=32, method_path = None,is_method_init=True):#w
    if method_path is None:
        for i, block in enumerate(transformer.transformer_blocks):
            attn = block.attn
            ff = block.ff
            # for attn set some attribute
            attn.step = 0
            attn.block_id = i
            attn.total_latency = 0.0
            attn.full_ops = 0
            attn.efficient_ops = 0
            attn.forward = types.MethodType(cuda_timer(efficient_attention_forward), attn)
            if is_method_init:
                attn.steps_method = ['NONE'] * steps
            attn.need_cache_output = [True] * steps
            attn.cached_output = None
            # for ff set some attribute
            if is_method_init:
                ff.steps_method = ['NONE'] * steps
            ff.need_cache_output = [True] * steps
            ff.full_ops = 0
            ff.efficient_ops = 0
            ff.block_id = i
            ff.step = 0
            ff.total_latency = 0.0
            ff.forward = types.MethodType(cuda_timer(efficient_ff_forward), ff)
            ff.cached_output = None
    else:
        with open(method_path, 'r') as f:
            import json
            saved_methods = json.loads(open(method_path).read())['methods']

            for methods, block in zip(saved_methods, transformer.transformer_blocks):
                # for attn
                attn = block.attn
                attn.steps_method = methods
                attn.step = 0
                attn.total_latency = 0.0
                attn.full_ops = 0
                attn.efficient_ops = 0
                attn.forward = types.MethodType(cuda_timer(efficient_attention_forward), attn)
                attn.need_cache_output = [True] * steps
                attn.cached_output = None
                # for ff
                ff = block.ff
                ff.steps_method = methods
                ff.need_cache_output = [True] * steps
                ff.full_ops = 0
                ff.efficient_ops = 0
                ff.total_latency = 0.0
                ff.step = 0
                ff.forward = types.MethodType(cuda_timer(efficient_ff_forward), ff)
                ff.cached_output = None

# after every calibration phase modules' steps should be reset
def calibration_reset(transformer, steps=32):
    for block in transformer.transformer_blocks:
        attn = block.attn
        ff = block.ff
        # for attn set some attribute
        attn.step = 0
        attn.total_latency = 0.0
        attn.need_cache_output = [True] * steps
        attn.cached_output = None
        # for ff set some attribute
        ff.need_cache_output = [True] * steps
        ff.step = 0
        ff.total_latency = 0.0
        ff.cached_output = None

def eval_reset(transformer, steps=32):
    for block in transformer.transformer_blocks:
        attn = block.attn
        ff = block.ff
        # for attn set some attribute
        attn.step = 0
        attn.need_cache_output = [True] * steps
        attn.cached_output = None
        # for ff set some attribute
        ff.need_cache_output = [True] * steps
        ff.step = 0
        ff.cached_output = None

def efficient_ff_forward(self, x):
    """
    Efficient feed-forward computation based on different methods:
    - TS: Use cached output
    - BS: Compute for half batch and mirror
    - NONE: Regular computation
    """
    method = self.steps_method[self.step]
    if 'TS' in method:
        self.step += 1
        return self.cached_output
    elif 'BS' in method:
        # Split batch computation
        x = self.ff(x[:x.shape[0]//2])
        batch_size, _ , _ = x.shape
        x = self.cached_output - self.cached_output[:batch_size] + x
        # Calculate residual between conditional and unconditional
        # out_cond = self.ff(x)
        # out_cond_res = self.cached_output[batch_size:] - self.cached_output[:batch_size]
        # out = torch.cat([out_cond, out_cond + out_cond_res], dim=0)
        if self.need_cache_output[self.step]:
            self.cached_output = x
        self.step += 1
        return x
    elif 'NONE' in method:
        out = self.ff(x)
        if self.need_cache_output[self.step]:
            self.cached_output = out
        self.step += 1
        return out
    else:
        raise NotImplementedError

def efficient_attention_forward(
    self,
    x: float, 
    mask: bool | None = None, 
    rope=None,  # rotary position embedding
    block_id=None,
    enable_flash_attn=True,
): 
    """
    Efficient attention computation with different strategies:
    - TS: Direct share from last step's output
    - BS: Compute conditional part only
    - Regular: Full attention computation
    """
    method = self.steps_method[self.step]

    # Whether to directly share output from the latest step (TS mechanism)
    if 'TS' in method:
        self.step += 1
        return self.cached_output

    # BS mechanism calculation
    # For BS mechanism, only calculate conditional part first    
    if 'BS' in method:
        # Exclude unconditional part
        x = x[:x.shape[0]//2]

    # Sample projections
    query = self.to_q(x)
    key = self.to_k(x)
    value = self.to_v(x)

    batch_size, seq_len, _ = x.shape

    # Apply rotary position embedding
    if rope is not None:
        freqs, xpos_scale = rope
        q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

        query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
        key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

    # Attention computation
    inner_dim = key.shape[-1]
    head_dim = inner_dim // self.heads
    query = query.view(batch_size, -1, self.heads, head_dim)
    key = key.view(batch_size, -1, self.heads, head_dim)
    value = value.view(batch_size, -1, self.heads, head_dim)

    x = flash_attn_func(query, key, value, causal=False)

    x = x.view(batch_size, seq_len, -1).to(query.dtype)
        
    # Linear projection
    x = self.to_out[0](x)
    
    # Dropout layer
    x = self.to_out[1](x)

    if mask is not None:
        mask = mask.unsqueeze(-1)
        x = x.masked_fill(~mask, 0.0)

    if 'BS' in method:
        x = self.cached_output - self.cached_output[:batch_size] + x
    
    if self.need_cache_output[self.step]:
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