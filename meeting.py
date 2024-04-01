import streamlit as st
import numpy as np
import whisper
from sklearn.cluster import AgglomerativeClustering
import librosa
import openai
import io
import os

# Assuming the OpenAI API key is set as an environment variable for security reasons
openai.api_key = "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"

if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a']

def divide_audio_chunks(audio_data, sr, chunk_duration_sec=300):
    samples_per_chunk = sr * chunk_duration_sec
    total_samples = len(audio_data)
    chunks = [audio_data[i:i + samples_per_chunk] for i in range(0, total_samples, samples_per_chunk)]
    return chunks


def process_audio_chunk(chunk, model_type, run_device='cpu'):
    whisper_model = whisper.load_model(model_type, run_device)
    wp_results = whisper_model.transcribe(chunk)

    # Simulated diarization logic
    # This part is a placeholder. In a real scenario, you'd use actual diarization techniques,
    # possibly involving machine learning models to distinguish between different speakers.
    num_segments = len(wp_results['segments'])
    for i, segment in enumerate(wp_results['segments']):
        # Example strategy: alternate speaker labels for demonstration purposes
        speaker_label = f"Speaker {i % 2 + 1}"
        segment['speaker'] = speaker_label

    return wp_results



def process_wav(audio_data, sr, speaker_number, model_type, run_device='cpu'):
    st.write("Initializing audio file processing...")
    progress_bar = st.progress(0)

    # Ensure audio data is resampled to 16kHz as Whisper expects
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000  # Update sample rate to reflect the new sample rate

    chunks = divide_audio_chunks(audio_data, sr)
    progress_bar.progress(10)

    all_results = []
    for i, chunk in enumerate(chunks):
        st.write(f"Processing chunk {i + 1}/{len(chunks)}...")
        chunk_results = process_audio_chunk(chunk, model_type, run_device)
        all_results.append(chunk_results)
        progress_bar.progress(10 + int(90 * ((i + 1) / len(chunks))))

    st.write("All chunks processed.")
    progress_bar.progress(100)
    return all_results


def login_ui():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "admin" and password == "password":
            st.session_state['authenticated'] = True
            st.experimental_rerun()
        else:
            st.sidebar.error("Authentication failed.")

def get_gpt_insights(text, model="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a highly knowledgeable assistant asked to provide insights based on a transcript."},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        if response.choices:
            return response.choices[0].message['content'].strip()
        else:
            return "No insights could be generated."
    except Exception as e:
        st.error(f"Failed to fetch insights from GPT: {str(e)}")
        return ""


def main_ui():
    st.markdown('<p class="big-font">Speaker Diarization Application</p>', unsafe_allow_html=True)
    file = st.file_uploader("Upload an audio file:", type=['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a'])
    num_speakers = st.number_input("Number of speakers:", min_value=2, max_value=100, value=2)
    model_list = ['tiny', 'small', 'base', 'medium']
    model_type = st.selectbox("Select model type:", model_list)

    if file is not None and st.button("Analyze"):
        with st.spinner('Processing...'):
            audio_data, sr = librosa.load(file, sr=None, mono=True)
            all_results = process_wav(audio_data, sr, num_speakers, model_type)

            # Aggregate transcripts and segments
            aggregated_transcript = ""
            all_segments = []

            for result in all_results:
                if 'text' in result:
                    aggregated_transcript += result['text'] + " "
                if 'segments' in result:
                    all_segments.extend(result['segments'])

            # Display the aggregated transcripts
            st.write("Aggregated Text:")
            st.write(aggregated_transcript)

            # Display the aggregated segments
            for seg in all_segments:
                speaker_label = seg.get('speaker', 'Unknown Speaker')
                seg_text = seg.get('text', '')
                start_time = seg.get('start', 'Unknown start time')
                end_time = seg.get('end', 'Unknown end time')
                st.write(f"{speaker_label}: {seg_text} (from {start_time} to {end_time} seconds)")

            # Generate and display GPT insights based on the aggregated transcript
            insights = get_gpt_insights(aggregated_transcript)
            if insights:
                st.write("Insights from GPT:")
                st.write(insights)
            else:
                st.write("No insights were provided.")

    else:
        st.write("Please upload an audio file and specify the number of speakers to start analysis.")


def main():
    if not st.session_state['authenticated']:
        login_ui()
    else:
        main_ui()

if __name__ == "__main__":
    main()
