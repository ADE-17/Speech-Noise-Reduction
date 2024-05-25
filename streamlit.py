import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sounddevice as sd
import soundfile as sf
import io
from scipy.io.wavfile import write
from denoiser_utils import reduce_noise, audio2text
from denoiser import pretrained
import whisper

denoiser_model = pretrained.master64().cpu()
whisper_model = whisper.load_model("base")

def record_audio(duration, rate=16000):
    audio_array = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype='float32')
    sd.wait()  
    return audio_array.flatten(), rate

def denoise_audio(audio_array, model):
    denoised_array = reduce_noise(model, audio_array)
    return denoised_array

def array_to_wav_bytes(audio_array, rate):
    wav_io = io.BytesIO()
    write(wav_io, rate, (audio_array * 32767).astype(np.int16))  
    return wav_io

def plot_spectrogram_subplot(original_audio, denoised_audio, rate):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    _, _, _, im1 = ax[0].specgram(original_audio, Fs=rate, cmap='viridis', NFFT=256)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].set_title('Original Spectrogram')
    fig.colorbar(im1, ax=ax[0], label='Intensity (dB)')
    
    _, _, _, im2 = ax[1].specgram(denoised_audio, Fs=rate, cmap='viridis', NFFT=256)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Frequency (Hz)')
    ax[1].set_title('Denoised Spectrogram')
    fig.colorbar(im2, ax=ax[1], label='Intensity (dB)')
    
    return fig


def main():
    st.set_page_config(page_title="Audio Denoiser", page_icon=":sound:", layout="wide")

    st.title("The NoiseFathers")

    st.header("Record your audio")
    duration = st.slider("Set Recording Duration (seconds):", min_value=1, max_value=30, value=5)
    record_button = st.button("Record")

    if record_button:
        st.write("Recording...")
        audio_array, rate = record_audio(duration)
        st.write("Recording finished.")

        denoised_array = denoise_audio(audio_array, denoiser_model)
        noise_array = audio_array - denoised_array
        
        st.subheader("Audio Playback")
        st.audio(array_to_wav_bytes(audio_array, rate), format="audio/wav", start_time=0)
        st.write("Processed Audio Playback")
        st.audio(array_to_wav_bytes(denoised_array, rate), format="audio/wav", start_time=0)
        st.write("Noise/Error Playback")
        st.audio(array_to_wav_bytes(noise_array, rate), format="audio/wav", start_time=0)

        if st.button("View Plots"):
            st.subheader("Plots")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(audio_array)
            ax[0].set_title('Original Audio')
            ax[1].plot(denoised_array)
            ax[1].set_title('Processed Audio')
            st.pyplot(fig)

        if st.button("View Spectrograms"):
            st.subheader("Spectrograms")
            spectrogram_fig = plot_spectrogram_subplot(audio_array, denoised_array, rate)
            # original_spectrogram_fig = plot_spectrogram(audio_array, rate)
            # denoised_spectrogram_fig = plot_spectrogram(denoised_array, rate)
            # st.pyplot(original_spectrogram_fig)
            st.pyplot(spectrogram_fig)


    st.header("Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        audio_array, rate = sf.read(uploaded_file, dtype='float32')
        if audio_array.ndim > 1:
            audio_array = audio_array[:, 0]  

        st.write("Uploaded audio file:", uploaded_file.name)

        denoised_array = denoise_audio(audio_array, denoiser_model)
        noise_array = audio_array - denoised_array
        
        st.subheader("Audio Playback")
        st.audio(array_to_wav_bytes(audio_array, rate), format="audio/wav", start_time=0)
        st.write("Processed Audio Playback")
        st.audio(array_to_wav_bytes(denoised_array, rate), format="audio/wav", start_time=0)
        st.write("Noise/Error Playback")
        st.audio(array_to_wav_bytes(noise_array, rate), format="audio/wav", start_time=0)

        if st.button("View Plots", key="upload"):
            st.subheader("Plots")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(audio_array)
            ax[0].set_title('Original Audio')
            ax[1].plot(denoised_array)
            ax[1].set_title('Processed Audio')
            st.pyplot(fig)

        if st.button("View Spectrograms", key="upload_spectrogram"):
            st.subheader("Spectrograms")
            spectrogram_fig = plot_spectrogram_subplot(audio_array, denoised_array, rate)
            # original_spectrogram_fig = plot_spectrogram(audio_array, rate)
            # denoised_spectrogram_fig = plot_spectrogram(denoised_array, rate)
            # st.pyplot(original_spectrogram_fig)
            st.pyplot(spectrogram_fig)

        if st.button("Generate caption", key="upload"):
            st.write("Generating caption...")
            subtitles, language = audio2text(denoised_array, whisper_model)
            st.write(f"Detected language: {language}")
            st.write(f"Caption: {subtitles}")

if __name__ == "__main__":
    main()
