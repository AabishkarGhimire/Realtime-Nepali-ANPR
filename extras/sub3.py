import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment

MAX_CHARACTERS_PER_LINE = 50  # Adjust this value based on your preference

def transcribe_audio(audio_file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, show_all=False)
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    return None

def create_subtitle_file(text, audio_duration, output_subtitle_file='output_subtitle.srt'):
    if text is None:
        return

    # Split the text into lines with a maximum number of characters
    lines = [text[i:i+MAX_CHARACTERS_PER_LINE] for i in range(0, len(text), MAX_CHARACTERS_PER_LINE)]

    # Create a SubRip (srt) subtitle file
    subtitle_file_content = ""
    
    # Initialize timestamp variables
    start_timestamp = 0

    for i, line in enumerate(lines):
        line_duration = len(line.split())
        end_timestamp = start_timestamp + line_duration

        subtitle_file_content += f"{i + 1}\n"
        subtitle_file_content += f"{convert_seconds(start_timestamp)} --> {convert_seconds(end_timestamp)}\n"
        subtitle_file_content += f"{line}\n\n"

        # Update start timestamp for the next line
        start_timestamp = end_timestamp

    with open(f'./extras/{output_subtitle_file}', 'w') as subtitle_file:
        subtitle_file.write(subtitle_file_content)

    print(f"Subtitle file '{output_subtitle_file}' created successfully.")

def convert_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},000"

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_audio_duration(audio_file_path):
    audio = AudioSegment.from_wav(audio_file_path)
    return len(audio) / 1000.0  # Convert milliseconds to seconds

def main():
    audio_file_path = './a.wav'  # Replace with the path to your audio file
    text = transcribe_audio(audio_file_path)

    if text:
        print("Transcription:")
        print(text)
        
        audio_duration = get_audio_duration(audio_file_path)

        create_subtitle_file(text, audio_duration)
        text_to_speech(text)
    else:
        print("Failed to transcribe the audio.")

if __name__ == "__main__":
    main()
