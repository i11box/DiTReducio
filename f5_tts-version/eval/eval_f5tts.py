import soundfile as sf
import torch

from f5_tts.infer.infer_cli import infer_process, load_model, load_vocoder, DiT
from eval.utils_eval import ensure_dir
from cached_path import cached_path 

def process_sentence_pair(src_info, tgt_info, model, vocoder, output_dir, speed, delta=None):
    """Process a pair of sentences and generate audio
    
    Returns:
        tuple: (inference_time(s), audio_duration(s))
    """
    ensure_dir(output_dir)
    output_path = f"{output_dir}/{tgt_info['id']}.flac"
    
    # Time only the infer_process
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    # Generate target sentence audio
    audio,final_sample_rate, _ = infer_process(
        ref_audio=src_info['audio_path'],
        ref_text=src_info['text'],
        gen_text=tgt_info['text'],
        model_obj=model,
        vocoder=vocoder,
        mel_spec_type='vocos',
        speed=speed,
        delta = delta
    )
    
    end_event.record()
    torch.cuda.synchronize()
    infer_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
    
    with open(output_path,"wb") as f:
        sf.write(f.name, audio, final_sample_rate)
    
    # Calculate audio duration (seconds)
    audio_duration = len(audio) / final_sample_rate
    
    return infer_time, audio_duration

def main():

    config = {
        "paths": {
            "lst_file": "data/LibriSpeech/librispeech_pc_test_clean_cross_sentence.lst",
            "output_dir": "data/LibriSpeech/test-clean_output",
            "ckpt_file": "",
            "vocab_file": ""
        },
        "model": {
            "name": "F5-TTS",
            "vocoder_name": "vocos",
        },
        "generation": {
            "speed": 1.0,
            "delta": 0.2
        }
    }
    
    # Override config with command line args
    lst_file = config["paths"]["lst_file"]
    output_dir = config["paths"]["output_dir"]
    model_name = config["model"]["name"]
    ckpt_file = config["paths"]["ckpt_file"]
    vocab_file = config["paths"]["vocab_file"]
    vocoder_name = config["model"]["vocoder_name"]
    speed = config["generation"]["speed"]
    delta = config["generation"]["delta"]

    # Load vocoder
    vocoder = load_vocoder(vocoder_name=vocoder_name)

    # Load model
    if model_name == "F5-TTS":
        model_cls = DiT
        model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        if ckpt_file == "":
            if vocoder_name == "vocos":
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base"
                ckpt_step = 1200000
                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
            elif vocoder_name == "bigvgan":
                repo_name = "F5-TTS"
                exp_name = "F5TTS_Base_bigvgan"
                ckpt_step = 1250000
                ckpt_file = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.pt"))
    else:
        print("Only testing F5-TTS model.")
        return
    
    model = load_model(model_cls, model_cfg, ckpt_file, 
                      mel_spec_type=vocoder_name, 
                      vocab_file=vocab_file)
    
    # Read lst file
    with open(lst_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # Process each sentence pair
    processed_pairs = 0
    total_audio_duration = 0.0
    total_infer_time = 0.0
    
    for i, line in enumerate(lines):
        try:
            # Parse line
            src_id, src_dur, src_text, tgt_id, tgt_dur, tgt_text = line.strip().split('\t')
            
            # Build audio paths
            src_audio_path = f"data/LibriSpeech/test-clean/{src_id.split('-')[0]}/{src_id.split('-')[1]}/{src_id}.flac"
            tgt_audio_path = output_dir
            
            # Prepare sentence info
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
            
            # Process this pair
            infer_time, audio_duration = process_sentence_pair(
                src_info, tgt_info, model, vocoder, tgt_audio_path, speed,delta
            )
            
            # Accumulate statistics
            total_infer_time += infer_time
            total_audio_duration += audio_duration
            processed_pairs += 1
            
            print(f"Processed pair {i}/{len(lines)}: {src_id} -> {tgt_id} (infer: {infer_time:.2f}s, audio: {audio_duration:.2f}s)")
            print(f'Current RTF: {total_infer_time/total_audio_duration:.4f}')
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing line {i}:\n{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            continue
    
    # Print statistics
    print(f'\nProcessed {processed_pairs} sentence pairs:')
    print(f'Total inference time: {total_infer_time:.2f} seconds')
    print(f'Total generated audio duration: {total_audio_duration:.2f} seconds')
    print(f'Real-time Factor (RTF): {total_infer_time/total_audio_duration:.4f}')

if __name__ == "__main__":
    main()
