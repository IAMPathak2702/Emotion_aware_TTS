import os
import librosa
from emotion_tts.config import PROCESSED_DATA_DIR, EMOTIONS, BASE_DIR

def create_metadata():
    metadata = []
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(PROCESSED_DATA_DIR, emotion)
        for file in os.listdir(emotion_dir):
            if file.endswith('.wav'):
                filepath = os.path.join(emotion_dir, file)
                duration = librosa.get_duration(filename=filepath)
                text = f"This is a {emotion.lower()} speech."
                metadata.append(f"{filepath}|{text}|{emotion}")
    
    with open(os.path.join(BASE_DIR, 'metadata.csv'), 'w') as f:
        for line in metadata:
            f.write(f"{line}\n")
    
    print(f"Metadata created with {len(metadata)} entries.")

if __name__ == "__main__":
    create_metadata()