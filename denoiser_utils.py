import torch
import sounddevice as sd
import numpy as np
from denoiser.dsp import convert_audio
import matplotlib.pyplot as plt
import soundfile as sf
from denoiser import pretrained
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import wave
import torch
from IPython import display as disp
import whisper

# model = pretrained.dns64().cpu()

def reduce_noise(model, audio):
    wav = torch.from_numpy(audio.astype(np.float32)).reshape(1,-1)
# wav = convert_audio(wav.cpu(), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
    # disp.display(disp.Audio(wav.data.cpu().numpy(), rate=model.sample_rate))
    # disp.display(disp.Audio(denoised.data.cpu().numpy(), rate=model.sample_rate))
    return np.array(denoised).reshape(-1)

# model = whisper.load_model("small")

def audio2text(audio, model):

    audio = whisper.pad_or_trim(audio)
    result = model.transcribe(audio)
    # print(result["text"])
    
    mel = whisper.log_mel_spectrogram(audio)

    _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")
    
    return result["text"], max(probs, key=probs.get)