# DiTReducio 
A calibration and acceleration tool for DiT-Based TTS models.

Audio demos for different thresholds are available in the `assets` folder.
## Supported Models
- F5-TTS
- MegaTTS 3

## Installation
```bash
git clone https://github.com/i11box/DiTReducio.git # clone the repository
```

- F5-TTS
  
  For F5-TTS, place the corresponding version files in the F5-TTS model root directory.
- MegaTTS 3
  
  For MegaTTS 3, place the corresponding version files in the `tts` folder of MegaTTS 3 model.

> Note: The paths in code should be adjusted according to your environment.

## Usage
```bash
python fast_cli.py # Launch
python fast_cli.py -q true -d <threshold> # Calibration, -q true enable calibration, -d <threshold> set threshold value
python fast_cli.py -d <threshold> # Acceleration
```
