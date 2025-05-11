import argparse
import codecs
import os
import re
import json

from datetime import datetime
from importlib.resources import files
from pathlib import Path

from f5_tts.calibration.util_calibration import threshold_q
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    device,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)

from f5_tts.calibration.hook import *
from f5_tts.calibration.util_calibration import seed_everything
parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-f",
    "--gen_file",
    type=str,
    help="The file with text to generate, will ignore --gen_text",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on",
)

parser.add_argument(
    "--calibration",
    "-q",
    type=bool,
    help="Calibration mode or not",
)

parser.add_argument(
    "--threshold",
    "-d",
    type=float,
    help="Compression thresholld, the larger the more compression",
)

parser.add_argument(
    "--seed",
    type=int,
    help="Random seed for reproducibility",
)

args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parameters

model = args.model or config.get("model", "F5TTS_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")

save_chunk = args.save_chunk or config.get("save_chunk", False)
remove_silence = args.remove_silence or config.get("remove_silence", False)
load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)
delta = args.threshold or config.get("threshold", None)
seed = args.seed or config.get("seed", 888)
seed_everything(seed) # TODO: only for experiment

if delta == 0:
    delta = None


# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "/root/.cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots/0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=True, local_path=vocoder_local_path, device=device
)


# load TTS model

model_cfg = OmegaConf.load(
    args.model_cfg or config.get("model_cfg", str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
)
model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch

repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"

if model != "F5TTS_Base":
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type

# override for previous models
if model == "F5TTS_Base":
    if vocoder_name == "vocos":
        ckpt_step = 1200000
    elif vocoder_name == "bigvgan":
        model = "F5TTS_Base_bigvgan"
        ckpt_type = "pt"
elif model == "E2TTS_Base":
    repo_name = "E2-TTS"
    ckpt_step = 1200000

if not ckpt_file:
    ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}"))

print(f"Using {model}...")
ema_model = load_model(
    model_cls, model_arc, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device
)

def ensure_dir(dir_path):
    """Ensure directory exists, create if not exists"""
    os.makedirs(dir_path, exist_ok=True)

def process_sentence_pair(src_info, tgt_info, output_dir):
    """Process a pair of sentences, generate and save audio
    
    Returns:
        tuple: (inference time (seconds), audio duration (seconds))
    """
    ensure_dir(output_dir)
    output_path = f"{output_dir}/{tgt_info['id']}.flac"
    
    ref_audio=src_info['audio_path']
    ref_text=src_info['text']
    gen_text=tgt_info['text']
    generated_audio_segments = []
    reg1 = r"(?=\[\w+\])"
    chunks = re.split(reg1, gen_text)
    reg2 = r"\[(\w+)\]"
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    for text in chunks:
        # recording flops
        hooks = []
        for block in ema_model.transformer.transformer_blocks:
            hook = block.attn.register_forward_pre_hook(calculate_flops_hook, with_kwargs=True)
            hook_ff = block.ff.register_forward_pre_hook(calculate_ff_flops_hook, with_kwargs=True)
            hooks.append(hook)
            hooks.append(hook_ff)
        
        # Generate target sentence audio

        audio_segment, final_sample_rate, spectragram = infer_process(
            ref_audio,
            ref_text,
            text,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=1.0,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device,
        )
        generated_audio_segments.append(audio_segment)
    
    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
    
    end_event.record()
    torch.cuda.synchronize()
    infer_time = start_event.elapsed_time(end_event) / 1000.0 
        
    for hook in hooks:
        hook.remove()
    
    total_full_ops = 0
    total_efficient_ops = 0
    total_full_ops_ff = 0
    total_efficient_ops_ff = 0
    total_attn_latency = 0.0
    total_ff_latency = 0.0
        
    for blocki, block in enumerate(ema_model.transformer.transformer_blocks):
        total_full_ops += block.attn.full_ops
        total_efficient_ops += block.attn.efficient_ops
        total_attn_latency += block.attn.total_latency
        total_ff_latency += block.ff.total_latency
        
        total_full_ops_ff += block.ff.full_ops
        total_efficient_ops_ff += block.ff.efficient_ops
    
    
    avg_full_ops = total_full_ops / len(ema_model.transformer.transformer_blocks)
    avg_efficient_ops = total_efficient_ops / len(ema_model.transformer.transformer_blocks)
    avg_full_ops_ff = total_full_ops_ff / len(ema_model.transformer.transformer_blocks)
    avg_efficient_ops_ff = total_efficient_ops_ff / len(ema_model.transformer.transformer_blocks)
    avg_attn_latency = total_attn_latency / len(ema_model.transformer.transformer_blocks)
    avg_ff_latency = total_ff_latency / len(ema_model.transformer.transformer_blocks)
    
    with open(output_path,"wb") as f:
        sf.write(f.name, final_wave, final_sample_rate)
    
    # Calculate audio duration (seconds)
    audio_duration = len(final_wave) / final_sample_rate
    
    calibration_reset(ema_model.transformer)
    
    return infer_time, audio_duration


# inference process


def main():

    # Load config
    config = {
        "paths": {
            "lst_file": "",  # your path, e.g. "data/data/LibriSpeech/librispeech_pc_test_clean_cross_sentence.lst"
            "output_dir": "", # your path, e.g. "data/data/LibriSpeech/test-clean_output"
            "ckpt_file": "", # your path
            "vocab_file": ""
        },
        "model": {
            "name": "F5-TTS",
            "vocoder_name": "vocos",
        }
    }
    
    lst_file = config["paths"]["lst_file"]
    output_dir = config["paths"]["output_dir"]

    with open(lst_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    processed_pairs = 0
    total_audio_duration = 0.0
    total_infer_time = 0.0

    if delta is not None:
        speedup(ema_model, steps=32, delta=delta)
    else:
        calibration_preparation(ema_model.transformer, steps=32)

    for i, line in enumerate(lines):
        src_id, src_dur, src_text, tgt_id, tgt_dur, tgt_text = line.strip().split('\t')
        
        src_audio_path = "" # your path, e.g. "data/data/LibriSpeech/test-clean/{src_id.split('-')[0]}/{src_id.split('-')[1]}/{src_id}.flac"
        tgt_audio_path = output_dir
        
        src_info = {
            'id': src_id,
            'duration': float(src_dur),
            'text': src_text,
            'audio_path': src_audio_path
        }
        
        tgt_info = {
            'id': tgt_id,
            'duration': float(tgt_dur),
            'text': tgt_text,
            'audio_path': tgt_audio_path
        }
        
        # Process this pair of sentences
        infer_time, audio_duration = process_sentence_pair(
            src_info, tgt_info, tgt_audio_path
        )
        
        if i > 1:
            total_infer_time += infer_time
            total_audio_duration += audio_duration
            processed_pairs += 1
            
        print(f"Processed pair {i}/{len(lines)}: {src_id} -> {tgt_id} (infer: {infer_time:.2f}s, audio: {audio_duration:.2f}s)")
    
    # Print statistics
    print(f'\nProcessed {processed_pairs} pairs of sentences:')
    print(f'Total inference time: {total_infer_time:.2f} seconds')
    print(f'Total audio duration: {total_audio_duration:.2f} seconds')
    print(f'RTF: {total_infer_time/total_audio_duration:.4f}')

if __name__ == "__main__":
    main()
