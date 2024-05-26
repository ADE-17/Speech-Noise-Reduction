import streamlit as st
from algorithms import ApplyRLS, RLSFilter
import soundfile as sf
import matplotlib.pyplot as plt

def apply_rls_page():
    st.markdown("<h1 style='color: red;'>🚀 Ultra-Fast Inference</h1>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        st.write("Uploaded audio file:", uploaded_file.name)

        audio_array, rate = sf.read(uploaded_file, dtype='float32')
        if audio_array.ndim > 1:
            audio_array = audio_array[:, 0]  

        st.subheader("Original Audio Playback")
        st.audio(audio_array, format="audio/wav", start_time=0)

        if st.button("Apply RLS Filter", key='apply'):
            st.write("Applying RLS Filter...")
            processed_audio = ApplyRLS(audio_array)
            st.subheader("Processed Audio Playback")
            st.audio(processed_audio, format="audio/wav", start_time=0)

        if st.button("View Impulse Response", key='im_res'):
            st.write("Loading RLS Filter...")
            rls_model = RLSFilter.load_model(r'C:\Users\ADE17\Desktop\Kickelhack\notebook\rls_1200.pkl')
            impulse_response = rls_model.weights  
            # st.write("Impulse Response:", impulse_response)
            plt.figure()
            plt.plot(impulse_response)
            plt.title("Impulse Response")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            st.pyplot()
            
        if st.button("Return to Main Page"):
            return

if __name__ == "__main__":
    apply_rls_page()
