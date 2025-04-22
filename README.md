# TriCAL 
A calibration and acceleration tool for TTS models.
## Supported Models
- F5-TTS
- - MegaTTS 3

## Installation
```bash
git clone https://github.com/i11box/TriCAL.git # clone the repository
```

- F5-TTS
  
  For F5-TTS, place the corresponding version files in the F5-TTS model root directory.
- MegaTTS 3
  
  For MegaTTS 3, place the corresponding version files in the `tts` folder of MegaTTS 3 model.

## Usage
Launch
```bash
python fast_cli.py
```

### Calibration Mode
```bash
python fast_cli.py -q true -d <threshold>
```
- `-q true`: Enable calibration
- `-d`: Set threshold value

### Acceleration Mode
```bash
python fast_cli.py -d <threshold> # after calibration
```