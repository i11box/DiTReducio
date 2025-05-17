import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 配置参数
OUTPUT_DIR = "assets/spectrograms"      # 输出频谱图的文件夹路径
IMG_SIZE = (8, 6)                # 图像大小（英寸）
DPI = 300                   # 分辨率
COLORMAP = 'magma'               # 颜色映射（可选 'inferno', 'viridis', 'plasma' 等）
INPUT_DIRS = ["f5-tts_en", "f5-tts_zh", "megatts3_en", "megatts3_zh"]  # 输入音频文件夹列表

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_threshold_value(filename, folder_name):
    # 提取文件名中的数值部分
    import re
    numbers = re.findall(r'\d+\.\d+', filename)
    if not numbers:
        return None
    f5_dict = {"0.05": 1, "0.1": 2, "0.15": 3, "0.2": 4, "0.25": 5, "0.3": 6}
    mega_dict = {"0.2": 1, "0.4": 2, "0.6": 3, "0.8": 4, "1.0": 5, "1.2": 6}
    
    # 根据不同文件夹使用不同的除数
    if folder_name.startswith("f5-tts"):
        threshold = f5_dict[numbers[0]]
    elif folder_name.startswith("megatts3"):
        threshold = mega_dict[numbers[0]]
    else:
        divisor = 1.0
    
    return f"T{threshold}"

def get_title_from_folder(folder_name, threshold):
    # 从文件夹名称生成标题前缀
    language = "EN" if folder_name.endswith("_en") else "ZH"
    
    if folder_name.startswith("f5-tts"):
        prefix = "F5-TTS"
    elif folder_name.startswith("megatts3"):
        prefix = "MegaTTS 3"
    else:
        prefix = folder_name
    if threshold:
        return f"{prefix} (Threshold={threshold}, {language})"
    else:
        return f"{prefix} (No Compression, {language})"

def audio_to_spectrogram(file_path, output_path, folder_name=None, filename=None):
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"[ERROR] Could not load {file_path}: {e}")
        return

    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))

    # 生成标题
    title = "Spectrogram"
    if folder_name and filename:
        threshold = extract_threshold_value(filename, folder_name)
        title = get_title_from_folder(folder_name, threshold)

    plt.figure(figsize=IMG_SIZE, dpi=DPI)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap=COLORMAP)
    # plt.colorbar(format="%+2.0f dB")
    plt.title(title, fontsize=14, pad=10)  # 设置标题字体大小和与图表的间距
    
    # 设置坐标轴标签的字体大小
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    
    # 设置刻度标签的字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()

def batch_generate_spectrograms(input_dir, output_dir, folder_name=None):
    ensure_dir(output_dir)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if not filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            continue

        name, ext = os.path.splitext(filename)
        output_file = os.path.join(output_dir, f"{name}.png")

        print(f"Processing: {filename} -> {output_file}")
        audio_to_spectrogram(file_path, output_file, folder_name, filename)

if __name__ == "__main__":
    for input_dir in INPUT_DIRS:
        output_dir = os.path.join(OUTPUT_DIR, input_dir)
        ensure_dir(output_dir)
        batch_generate_spectrograms(f"assets/{input_dir}", output_dir, input_dir)