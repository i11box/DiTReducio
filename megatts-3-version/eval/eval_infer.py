import json
import os
import argparse
import librosa
import numpy as np
import torch

from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
from langdetect import detect as classify_language
from pydub import AudioSegment
import pyloudnorm as pyln

from tts.calibration.hook import *
from tts.calibration.util_calibration import threshold_dbscan, seed_everything

from tts.modules.ar_dur.commons.nar_tts_modules import LengthRegulator
from tts.frontend_function import g2p, align, make_dur_prompt, dur_pred, prepare_inputs_for_dit
from tts.utils.audio_utils.io import save_wav, to_wav_bytes, convert_to_wav_bytes, combine_audio_segments
from tts.utils.commons.ckpt_utils import load_ckpt
from tts.utils.commons.hparams import set_hparams, hparams
from tts.utils.text_utils.text_encoder import TokenTextEncoder
from tts.utils.text_utils.split_text import chunk_text_chinese, chunk_text_english
from tts.utils.commons.hparams import hparams, set_hparams
from tts.fast_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav

import wandb
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ensure_dir(dir_path):
    """Ensure directory exists, create if not exists"""
    os.makedirs(dir_path, exist_ok=True)

def process_sentence_pair(src_info, tgt_info, output_dir):

    wav_path=src_info['audio_path']
    input_text=tgt_info['text']
    audio_duration = tgt_info['duration']
    ensure_dir(output_dir)
    output_path = f"{output_dir}/{tgt_info['id']}.flac"
    with open(wav_path, 'rb') as file:
        file_content = file.read()
        
    resource_context = infer_ins.preprocess(file_content)

    hooks = []
    for block in infer_ins.dit.encoder.layers:
        hook = block.attention.register_forward_pre_hook(calculate_flops_hook, with_kwargs=True)
        hook_ff = block.feed_forward.register_forward_pre_hook(calculate_ff_flops_hook, with_kwargs=True)
        hooks.append(hook)
        hooks.append(hook_ff)

    wav_bytes = infer_ins.forward(resource_context, input_text, time_step=time_step, p_w=p_w, t_w=t_w)

    for hook in hooks:
        hook.remove()
    
    total_full_ops = 0
    total_efficient_ops = 0
    total_attn_latency = 0.0
    total_ff_latency = 0.0
    
    for blocki, block in enumerate(infer_ins.dit.encoder.layers):
        total_full_ops += block.attention.full_ops
        total_efficient_ops += block.attention.efficient_ops
        total_attn_latency += block.attention.total_latency
        total_ff_latency += block.feed_forward.total_latency
        
    avg_full_ops = total_full_ops / len(infer_ins.dit.encoder.layers)
    avg_efficient_ops = total_efficient_ops / len(infer_ins.dit.encoder.layers)
    # avg_full_ops_ff = total_full_ops_ff / len(infer_ins.dit.encoder.layers)
    # avg_efficient_ops_ff = total_efficient_ops_ff / len(infer_ins.dit.encoder.layers)
    avg_attn_latency = total_attn_latency / len(infer_ins.dit.encoder.layers) * 1000
    avg_ff_latency = total_ff_latency / len(infer_ins.dit.encoder.layers) * 1000
    
    wandb.log({
        "Average Attn Latency": avg_attn_latency,
        "Average FF Latency": avg_ff_latency,
        "Average Attn&&FF Ops percent": avg_efficient_ops / avg_full_ops,
    })

    save_wav(wav_bytes, output_path)
    calibration_reset(infer_ins.dit.encoder)

    return audio_duration

if __name__ == '__main__':
    
    # Load config
    config = {
        "paths": {
            "lst_file": "data/LibriSpeech/librispeech_pc_test_clean_cross_sentence.lst",
            "output_dir": "data/LibriSpeech/test-clean_output",
        }, 
        "params": {
            "time_step": 32,
            "p_w": 1.6,
            "t_w": 2.5,
        }
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', '-d', type=float, default=None, help='Compression Threshold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    time_step, p_w, t_w = config["params"]["time_step"], config["params"]["p_w"], config["params"]["t_w"]
    delta, seed= args.delta, args.seed
    if delta == 0:
        delta = None
    seed_everything(seed) # TODO: only for experiment
    
    wandb.init(project="MegaTTS3-Evaluation-new", 
               name=f"LibriSpeech-Test-Clean_{delta}",
               group=f"{seed}"
               )
    
    lst_file = config["paths"]["lst_file"]
    output_dir = config["paths"]["output_dir"]

    with open(lst_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_pairs = 0
    total_audio_duration = 0.0
    total_infer_time = 0.0

    infer_ins = MegaTTS3DiTInfer()
    if delta is not None:
        speedup(infer_ins, steps = time_step, delta=delta)
    else:
        calibration_preparation(infer_ins.dit.encoder, steps = time_step)
        
    for i, line in enumerate(lines):
        src_id, src_dur, src_text, tgt_id, tgt_dur, tgt_text = line.strip().split('\t')
        
        tgt_audio_path = output_dir
        
        speaker_id = src_id.split('-')[0]
        src_audio_path = f"data/LibriSpeech/test-clean/{src_id.split('-')[0]}/{src_id.split('-')[1]}/{src_id}.flac"
        src_info = {
            'audio_path': src_audio_path
        }
        
        tgt_info = {
            'id': tgt_id,
            'text': tgt_text,
            'audio_path': tgt_audio_path,
            'duration': float(tgt_dur)
        }
        
        # Process this pair of sentences
        audio_duration = process_sentence_pair(
            src_info, tgt_info, tgt_audio_path
        )
        
        infer_time = infer_ins.dit.infer_times[i]
        processed_pairs += 1
            
        print(f"[INFO]:Processed pair {i}/{len(lines)}: {src_id} -> {tgt_id} (infer: {infer_time:.2f}s, audio: {audio_duration:.2f}s)")
    
    # Print statistics
    total_infer_time = sum(infer_ins.infer_times)
    print(f'\nProcessed {processed_pairs} pairs of sentences:')
    print(f'Total inference time: {total_infer_time:.2f} seconds')
    print(f'Total audio duration: {total_audio_duration:.2f} seconds')
    print(f'RTF: {total_infer_time/total_audio_duration:.4f}')