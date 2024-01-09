import speech_recognition as sr
import pyttsx3

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

def main():
    audio_file_path = './a.wav'  # Replace with the path to your audio file
    text = transcribe_audio(audio_file_path)

    if text:
        print("Transcription:")
        print(text)
        
        create_subtitle_file(text)
        text_to_speech(text)
    else:
        print("Failed to transcribe the audio.")

if __name__ == "__main__":
    main()
