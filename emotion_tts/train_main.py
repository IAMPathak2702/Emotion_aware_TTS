import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
import logging
import torch.multiprocessing as mp

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory structure and hyperparameters
BASE_DIR = "./"
DATA_DIR = os.path.join(BASE_DIR, 'emov-DB')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'processed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

EMOTIONS = ['Neutral', 'Sleepy', 'Angry', 'Disgusted', 'Amused']
EMOTION_FILE_COUNTS = {
    'Neutral': 373, 'Sleepy': 520, 'Angry': 317, 'Disgusted': 347, 'Amused': 309
}

HPARAMS = {
    'training_files': os.path.join(BASE_DIR, 'metadata.csv'),
    'val_files': os.path.join(BASE_DIR, 'metadata.csv'),  # Use a separate validation set in practice
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'sampling_rate': 22050,
    'filter_length': 1024,
    'hop_length': 256,
    'win_length': 1024,
    'n_mel_channels': 80,
    'mel_fmin': 0.0,
    'mel_fmax': 8000.0,
    'EMOTIONS': EMOTIONS
}

class EmotionalSpeechDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        logger.info(f"Initializing EmotionalSpeechDataset with csv_file={csv_file}, root_dir={root_dir}")
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.emotion_to_id = {emotion: i for i, emotion in enumerate(EMOTIONS)}
        logger.info(f"Dataset initialized with {len(self.metadata)} samples")
        logger.debug(f"Emotion to ID mapping: {self.emotion_to_id}")
        logger.debug(f"CSV columns: {self.metadata.columns}")

    def __getitem__(self, idx):
        logger.debug(f"Getting item {idx}")
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.root_dir, row['audio_file'])  # Adjust 'audio_file' to match your column name
        text = row['text']  # Adjust 'text' to match your column name
        emotion = row['emotion']  # Adjust 'emotion' to match your column name

        try:
            audio, _ = librosa.load(audio_path, sr=HPARAMS['sampling_rate'])
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=HPARAMS['sampling_rate'],
                n_mels=HPARAMS['n_mel_channels'],
                fmin=HPARAMS['mel_fmin'],
                fmax=HPARAMS['mel_fmax']
            )
            mel = librosa.power_to_db(mel, ref=np.max)

            return {
                'text': torch.LongTensor(self.text_to_sequence(text)),
                'mel': torch.FloatTensor(mel),
                'emotion': torch.LongTensor([self.emotion_to_id[emotion]])
            }
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            raise

    def text_to_sequence(self, text):
        # Implement text to sequence conversion here
        sequence = [ord(c) for c in text.lower()]
        logger.debug(f"Text to sequence: '{text}' -> {sequence}")
        return sequence

class EmotionalTacotron2(nn.Module):
    def __init__(self, tacotron2_model, num_emotions, emotion_embed_dim):
        super(EmotionalTacotron2, self).__init__()
        logger.info(f"Initializing EmotionalTacotron2 with num_emotions={num_emotions}, emotion_embed_dim={emotion_embed_dim}")
        self.tacotron2 = tacotron2_model
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embed_dim)
        
        old_embed_dim = self.tacotron2.embedding.embedding_dim
        self.new_embedding = nn.Embedding(
            self.tacotron2.embedding.num_embeddings,
            old_embed_dim + emotion_embed_dim
        )
        logger.info(f"New embedding layer created with input dim={self.tacotron2.embedding.num_embeddings} and output dim={old_embed_dim + emotion_embed_dim}")

    def forward(self, text, emotion):
        logger.debug(f"EmotionalTacotron2 forward pass: text shape={text.shape}, emotion shape={emotion.shape}")
        emotion_embed = self.emotion_embedding(emotion)
        text_embed = self.new_embedding(text)
        
        combined_embed = torch.cat((text_embed, emotion_embed.unsqueeze(1).expand(-1, text_embed.size(1), -1)), dim=-1)
        
        encoder_outputs = self.tacotron2.encoder(combined_embed)
        
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.tacotron2.decoder(
            encoder_outputs,
            memory=None,
            processed_memory=None,
            attention_weights=None
        )
        
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

def train(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        text = batch['text'].to(device)
        mel_target = batch['mel'].to(device)
        emotion = batch['emotion'].to(device)

        mel_output, mel_output_postnet, _, _ = model(text, emotion)

        loss = criterion(mel_output, mel_target) + criterion(mel_output_postnet, mel_target)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def main():
    logger.info("Script started")

    logger.info(f"Hyperparameters: {HPARAMS}")
    logger.info(f"Directories: BASE_DIR={BASE_DIR}, DATA_DIR={DATA_DIR}, PROCESSED_DATA_DIR={PROCESSED_DATA_DIR}, MODEL_DIR={MODEL_DIR}")

    # Load the pre-trained Tacotron2 model
    logger.info("Loading pre-trained Tacotron2 model")
    tacotron2_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
    tacotron2_model.eval()
    logger.info("Pre-trained Tacotron2 model loaded")

    # Create the emotional Tacotron2 model
    emotional_tacotron2 = EmotionalTacotron2(tacotron2_model, num_emotions=len(EMOTIONS), emotion_embed_dim=64)
    logger.info("EmotionalTacotron2 model created")

    # Prepare dataset and dataloader
    logger.info("Preparing dataset and dataloader")
    try:
        dataset = EmotionalSpeechDataset(HPARAMS['training_files'], DATA_DIR)
        dataloader = DataLoader(dataset, batch_size=HPARAMS['batch_size'], shuffle=True, num_workers=4)
        logger.info(f"Dataset and dataloader prepared with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(emotional_tacotron2.parameters(), lr=HPARAMS['learning_rate'])
    logger.info("Loss function and optimizer set up")

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    emotional_tacotron2.to(device)

    try:
        for epoch in range(HPARAMS['epochs']):
            logger.info(f"Starting epoch {epoch+1}/{HPARAMS['epochs']}")
            avg_loss = train(emotional_tacotron2, dataloader, criterion, optimizer, device, epoch)
            logger.info(f"Epoch {epoch+1}/{HPARAMS['epochs']}, Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(MODEL_DIR, f'emotional_tacotron2_checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': emotional_tacotron2.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise

    # Save the final model
    final_model_path = os.path.join(MODEL_DIR, 'emotional_tacotron2_final.pth')
    torch.save(emotional_tacotron2.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

    logger.info("Training completed")

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()