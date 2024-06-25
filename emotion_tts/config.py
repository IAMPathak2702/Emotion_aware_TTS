import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'emov-DB')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

EMOTIONS = ['Neutral', 'Sleepy', 'Angry', 'Disgusted', 'Amused']
EMOTION_FILE_COUNTS = {
    'Neutral': 373,
    'Sleepy': 520,
    'Angry': 317,
    'Disgusted': 347,
    'Amused': 309
}

HPARAMS = {
    'training_files': os.path.join(BASE_DIR, 'metadata.csv'),
    'val_files': os.path.join(BASE_DIR, 'metadata.csv'),  # Use a separate validation set in practice
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'sampling_rate': 22050,
    'filter_length': 1024,
    'hop_length': 256,
    'win_length': 1024,
    'n_mel_channels': 80,
    'mel_fmin': 0.0,
    'mel_fmax': 8000.0,
    'EMOTIONS': EMOTIONS  # Add this line
}