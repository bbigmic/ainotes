import os
import whisper
import requests
from pydub import AudioSegment
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
import warnings
import subprocess
import torch
import streamlit as st
import tempfile
import time
from datetime import datetime
import yt_dlp
from openai import OpenAI
from dotenv import load_dotenv

# Wczytaj zmienne z pliku .env
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Konfiguracja globalna
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")
torch.serialization.default_restore_location = lambda storage, loc: storage
torch.serialization.weights_only = True

# MODEL_URL = "http://localhost:11434/api/generate"  # Ollama API endpoint
# MODEL_NAME = "mistral"

SUPPORTED_AUDIO = (".wav", ".mp3", ".m4a", ".flac")
SUPPORTED_VIDEO = (".mp4", ".mov", ".avi", ".mkv")
MAX_FILE_SIZE_MB = 500  # Maksymalny rozmiar pliku w MB

def is_valid_file(file_path):
    try:
        command = ["ffmpeg", "-v", "error", "-i", file_path, "-f", "null", "-"]
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def download_video(url):
    output_path = "downloaded_video.mp4"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        raise ValueError(f"Failed to download video: {e}")

def convert_to_wav(file_path):
    print(f"Converting file: {file_path}")
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    if not is_valid_file(file_path):
        raise ValueError(f"File {file_path}  is corrupted or unsupported.")
    
    if file_ext in SUPPORTED_AUDIO:
        audio = AudioSegment.from_file(file_path)
        output_path = "converted_audio.wav"
        audio.export(output_path, format="wav")
        return output_path
    elif file_ext in SUPPORTED_VIDEO:
        output_path = "extracted_audio.wav"
        try:
            ffmpeg_extract_audio(file_path, output_path)
        except Exception as e:
            raise ValueError(f"Failed to extract audio: {e}")
        return output_path
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

def transcribe_audio(audio_path, language):
    print("Transcribing audio...")
    try:
        model = whisper.load_model("large")
        result = model.transcribe(audio_path, language=language if language != "auto" else None)
        return result['text']
    except Exception as e:
        return f"Transcription error: {e}"

def analyze_transcription(transcription, language):
    print("Analyzing key conversation points...")
    
    prompts = {
        "pl": f"""
        Oto zapis rozmowy. Podziel najważniejsze informacje na 3 sekcje:
        1. **Najważniejsze ustalenia**
        2. **Zadania do wykonania**
        3. **Dodatkowe notatki**
        
        Transkrypcja:
        {transcription}
        """,
        "en": f"""
        Here is the transcription of the conversation. Divide the key information into three sections:
        1. **Key Decisions**
        2. **Tasks to Complete**
        3. **Additional Notes**
        
        Transcription:
        {transcription}
        """,
        "de": f"""
        Hier ist die Transkription des Gesprächs. Teilen Sie die wichtigsten Informationen in drei Abschnitte auf:
        1. **Wichtige Entscheidungen**
        2. **Zu erledigende Aufgaben**
        3. **Zusätzliche Notizen**
        
        Transkription:
        {transcription}
        """,
        "fr": f"""
        Voici la transcription de la conversation. Divisez les informations clés en trois sections :
        1. **Décisions importantes**
        2. **Tâches à accomplir**
        3. **Notes supplémentaires**
        
        Transcription :
        {transcription}
        """
    }

   # Domyślnie używamy języka angielskiego, jeśli dany język nie jest obsługiwany
    prompt = prompts.get(language, prompts["en"])

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Możemy zmienić na gpt-3.5-turbo dla niższych kosztów
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"

# Funkcja analizy z niestandardowym promptem
def analyze_with_custom_prompt(transcription, original_notes, custom_prompt):
    print("Analyzing with a custom prompt...")
    
    combined_prompt = f"""

    Perform the following task: "{custom_prompt}" based on the transcription.

    **Transkrypcja:**
    {transcription}

    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": combined_prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"


# Funkcja do zapisywania transkrypcji i notatek w jednym pliku (tworzymy raz)
def save_transcription_and_notes(transcription, notes):
    filename = f"meeting_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("📌 **Transcription:**\n")
        f.write(transcription + "\n\n")
        f.write("📝 **Notes:**\n")
        f.write(notes)
    
    return file_path



def main():
    st.title("Audio/Video Transcription and Note Generation")
    
    # Session variables for new prompts
    if "transcription" not in st.session_state:
        st.session_state.transcription = None
    if "notes" not in st.session_state:
        st.session_state.notes = None
    if "custom_notes" not in st.session_state:
        st.session_state.custom_notes = None

    language = st.selectbox("Select transcription language", ["auto", "pl", "en", "de", "fr"])
    video_url = st.text_input("Paste YouTube or Instagram link")
    uploaded_file = st.file_uploader("Select an audio or video file", type=list(SUPPORTED_AUDIO) + list(SUPPORTED_VIDEO))

    if video_url:
        st.info("Processing video...")
        file_path = download_video(video_url)
    elif uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.read())
            file_path = temp_file.name
    else:
        return
    
    if os.path.getsize(file_path) > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error("The file is too large! The maximum size is 500 MB.")
        os.remove(file_path)
        return
    
    progress = st.progress(0)
    status_placeholder = st.empty()
    try:
        status_placeholder.text("Converting file to WAV format...")
        progress.progress(25)
        audio_path = convert_to_wav(file_path)
        
        status_placeholder.text("Transcribing audio...")
        progress.progress(50)
        st.session_state.transcription = transcribe_audio(audio_path, language)
        st.text_area("Transcription", st.session_state.transcription, height=300)
        
        status_placeholder.text("Analyzing key conversation points...")
        progress.progress(75)
        st.session_state.notes = analyze_transcription(st.session_state.transcription, language)
        st.text_area("Notes", st.session_state.notes, height=300)
        
        status_placeholder.text("Saving transcription and notes...")
        progress.progress(100)

        # Check if the file already exists; if not, save it
        if "summary_file" not in st.session_state:
            st.session_state.summary_file = save_transcription_and_notes(
                st.session_state.transcription, st.session_state.notes
            )

        # Provide a download button but do NOT regenerate the file!
        with open(st.session_state.summary_file, "rb") as file:
            st.download_button(
                "📥 Download Transcription and Notes",
                data=file,
                file_name=os.path.basename(st.session_state.summary_file),
                mime="text/plain"
            )

        status_placeholder.success("Task successfully completed! ✅")

        # Custom prompt section
        st.header("Customize the Prompt and Generate New Notes")
        custom_prompt = st.text_area("Enter a custom prompt", height=150)
        
        if st.button("Generate Notes Based on Custom Prompt"):
            if custom_prompt.strip():
                progress.progress(0)
                status_placeholder.text("Generating notes with custom prompt...")
                progress.progress(85)
                
                st.session_state.custom_notes = analyze_with_custom_prompt(
                    st.session_state.transcription,
                    st.session_state.notes,
                    custom_prompt
                )

                progress.progress(100)
                status_placeholder.success("New notes have been generated! ✅")
            else:
                st.error("Please enter a custom prompt!")

        # New section for displaying notes generated with the custom prompt
        if st.session_state.custom_notes:
            st.header("New Notes Based on Custom Prompt")
            st.text_area("New Notes", st.session_state.custom_notes, height=300)
        
    except Exception as e:
        status_placeholder.error(f"Error: {e}")
    finally:
        os.remove(file_path)
        progress.empty()

if __name__ == "__main__":
    main()
