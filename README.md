# Emotional Speech Synthesis with Tacotron 2

This project implements an emotional speech synthesis model based on Tacotron 2. It extends the original Tacotron 2 architecture to incorporate emotion embeddings, allowing for the generation of speech with different emotional tones.

## Project Overview

The main components of this project are:

1. A custom dataset class (`EmotionalSpeechDataset`) for loading and preprocessing emotional speech data.
2. A modified Tacotron 2 model (`EmotionalTacotron2`) that includes emotion embeddings.
3. A training script that fine-tunes the model on emotional speech data.

## Features

- Supports multiple emotions: Neutral, Sleepy, Angry, Disgusted, Amused
- Uses mel spectrograms as audio representations
- Implements logging for better tracking of the training process
- Includes checkpointing to save model progress

## Dependencies

- Python 3.x
- PyTorch
- torchaudio
- pandas
- numpy
- librosa
- logging

## Setup

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install the required dependencies:
   ```
   pip install torch torchaudio pandas numpy librosa
   ```

3. Prepare your data:
   - Place your audio files in the `emov-DB` directory
   - Create a `metadata.csv` file with columns: audio_file_path, text_transcript, emotion

4. Create necessary directories:
   ```
   mkdir processed_data saved_models
   ```

## Usage

To train the model, run the main script:

```
python train.py
```

The script will:
1. Load and preprocess the data
2. Initialize the EmotionalTacotron2 model
3. Train the model for the specified number of epochs
4. Save checkpoints every 5 epochs and the final model

## Model Architecture

The `EmotionalTacotron2` model extends the original Tacotron 2 architecture by:
1. Adding an emotion embedding layer
2. Modifying the encoder to accept combined text and emotion embeddings

## Logging

The script uses Python's `logging` module to track the training process. Logs are printed to the console and can be redirected to a file if needed.

## Future Improvements

- Implement a separate validation set for better model evaluation
- Add audio samples generation using the trained model
- Implement early stopping to prevent overfitting
- Add data augmentation techniques to improve model generalization

## Acknowledgements

This project uses the pre-trained Tacotron 2 model from NVIDIA's Deep Learning Examples repository.
