import os
import shutil
from emotion_tts.config import DATA_DIR, PROCESSED_DATA_DIR, EMOTIONS

def prepare_dataset():
    for emotion in EMOTIONS:
        src_dir = os.path.join(DATA_DIR, f'bea_{emotion}')
        dst_dir = os.path.join(PROCESSED_DATA_DIR, emotion)
        os.makedirs(dst_dir, exist_ok=True)
        
        for file in os.listdir(src_dir):
            if file.endswith('.wav'):
                shutil.copy(os.path.join(src_dir, file), dst_dir)
    
    print("Dataset preparation completed.")

if __name__ == "__main__":
    prepare_dataset()