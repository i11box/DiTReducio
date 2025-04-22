# This script comes from the F5-TTS repository and has been modified.
# Evaluate with Librispeech test-clean, ~3s prompt to generate 4-10s audio (the way of valle/voicebox evaluation)

import argparse
import sys
import os

sys.path.append(os.getcwd())

import numpy as np

from f5_tts.eval.utils_eval import (
    get_librispeech_test,
    run_asr_wer,
    run_sim,
)


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "--task",
    type=str,
    default="sim",
    help="task to evaluate, sim or wer",
)
parser.add_argument(
    "-d",
    "--delta",
    type=str,
    default="None",
    help="delta value for speedup, if applicable",
)

eval_task = parser.parse_args().task  # sim | wer
delta = parser.parse_args().delta  
if delta == 0:
    delta = None
lang = "en"
metalst = "" # your metalist path, the lst file can be found in F5-TTS repo
librispeech_test_clean_path = ""  # test-clean path
gen_wav_dir = ""  # generated wavs

gpus = [0]
test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path)

## In LibriSpeech, some speakers utilized varying voice characteristics for different characters in the book,
## leading to a low similarity for the ground truth in some cases.
# test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth = True)  # eval ground truth

local = True
if local:  # use local custom checkpoint dir
    asr_ckpt_dir = ""  # your ASR checkpoint directory
else:
    asr_ckpt_dir = ""  # auto download to cache dir

wavlm_ckpt_dir = ""  # your WavLM checkpoint directory


# --------------------------- WER ---------------------------

if eval_task == "wer":
    wers = []
    for rank, sub_test_set in test_set:
        wers_ = run_asr_wer((rank, lang, sub_test_set, asr_ckpt_dir))
        wers.extend(wers_)

    wer = round(np.mean(wers) * 100, 3)
    
    print(f"\nTotal {len(wers)} samples")
    print(f"WER      : {wer}%")

# --------------------------- SIM ---------------------------

if eval_task == "sim":
    sim_list = []

    for rank, sub_test_set in test_set:
        sim_ = run_sim((rank, sub_test_set, wavlm_ckpt_dir))
        sim_list.extend(sim_)

    total_sim = 0.0
    for s in sim_list:
        total_sim += s["sim"] # only use sim score

    sim = round(total_sim / len(sim_list), 3)

    print(f"\nTotal {len(sim_list)} samples")
    print(f"SIM      : {sim}")
