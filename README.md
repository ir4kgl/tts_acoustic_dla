# Text-to-Speech project

This repository contains TTS project done as a part of homework #3 for the DLA course at the CS Faculty of HSE. See [wandb report](https://wandb.ai/crimsonsparrow048/tts_acoustic/reports/TTS-Report--Vmlldzo2NTg3NTk0?accessToken=22nd1wlum50hn73n207ddmvwsghltxzraq8vz25nfvjjtnickbc2xnyidpncjlm1). 

## Installation guide

Clone this repository. Move to corresponding folder and install required packages:

```shell
git clone https://github.com/ir4kgl/tts_acoustic_dla
cd tts_acoustic_dla
pip install -r ./requirements.txt
```


## Waveglow model upload

This project contains acoustic part of TTS only (text to melspec). I do not implement vocoder part here and use trained waveglow model for evaluation (same model used in seminar). So you can run the following code to upload waveglow model and place it in correct directory:  

```shell
gdown https://drive.google.com/file/d/1Ojx_cEozcL4Jo6e4GbMONVQ-DLp5x_1L/view?usp=sharing
mv waveglow_256channels_ljs_v2.pt hw_tts/waveglow/pretrained_model/waveglow_256channels.pt
```

Also note that `waveclow` and `audio` folders in this repo is not my work, I used them like in seminar to complete preprocessing of LJSpeech dataset and generating final audios out of my Fastpeech model`s predictions. 


## Checkpoint

To download the final checkpoint run 

```shell
python3 download_checkpoint.py
```

it will download final checkpoint in `checkpoints/final_run` folder.

## Run train

To train model simply run

```shell
python3 train.py --c config.json --r CHECKPOINT.pth
```

where `config.json` is configuration file with all data, model and trainer parameters and `CHECKPOINT.pth` is an optional argument to continue training starting from a given checkpoint. 

Configuration of my final experiment you can find in the file `configs/final_run_config.json`.


Please note that you have to place LJSPeech dataset in current folder to make everything work fine. Also note that LJSpeech dataset should contain `mels/`, `pitch/` and `energy/` folders. Scripts for energiy and pitch calculations are attached (see the report).


