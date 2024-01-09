import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment

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

def create_subtitle_file(text, output_subtitle_file='output_subtitle.srt'):
    if text is None:
        return

    # Create a SubRip (srt) subtitle file
    subtitle_file_content = "1\n"
    subtitle_file_content += f"{convert_seconds(0)} --> {convert_seconds(len(text.split()))}\n"
    subtitle_file_content += f"{text}\n"

    with open(output_subtitle_file, 'w') as subtitle_file:
        subtitle_file.write(subtitle_file_content)

    print(f"Subtitle file '{output_subtitle_file}' created successfully.")

def save_output_audio(text, original_audio_path, output_audio_path='output_audio.wav'):
    if text is None:
        return

    # Synthesize speech using pyttsx3
    engine = pyttsx3.init()
    engine.save_to_file(text, output_audio_path)
    engine.runAndWait()

    # Load original audio and output audio using pydub
    original_audio = AudioSegment.from_wav(original_audio_path)
    output_audio = AudioSegment.from_wav(output_audio_path)

    # Match the frame rate of the output audio to the original audio
    output_audio = output_audio.set_frame_rate(original_audio.frame_rate)

    # Save the adjusted output audio
    output_audio.export(output_audio_path, format="wav")

    print(f"Output audio file '{output_audio_path}' saved successfully.")

def convert_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},000"

def main():
    original_audio_path = './a.wav'  # Replace with the path to your original audio file
    audio_file_path = './a.wav'  # Replace with the path to your audio file
    text = transcribe_audio(audio_file_path)

    if text:
        print("Transcription:")
        print(text)

        create_subtitle_file(text)
        save_output_audio(text, original_audio_path)
    else:
        print("Failed to transcribe the audio.")

if __name__ == "__main__":
    main()
