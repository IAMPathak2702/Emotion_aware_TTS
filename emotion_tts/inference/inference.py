import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from emotion_tts.model.tacotron2_emotion import Tacotron2Emotion
from emotion_tts.config import HPARAMS, EMOTIONS, MODEL_DIR
from emotion_tts.utils.text_processing import text_to_sequence

def plot_mel_spectrogram(mel_spectrogram, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def save_mel_spectrogram(mel_spectrogram, filename):
    np.save(filename, mel_spectrogram.cpu().numpy())

def load_model(checkpoint_path):
    model = Tacotron2Emotion(HPARAMS)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def inference(model, text, emotion):
    sequence = torch.LongTensor(text_to_sequence(text, ['english_cleaners']))[None, :]
    emotion_id = torch.LongTensor([EMOTIONS.index(emotion)])[None, :]
    
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference((sequence, emotion_id))
    
    return mel_outputs, mel_outputs_postnet, alignments

def load_waveglow(waveglow_path):
    waveglow = torch.load(waveglow_path)
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.eval()
    return waveglow.to('cuda' if torch.cuda.is_available() else 'cpu')

def vocoder_inference(waveglow, mel_spectrogram):
    mel = mel_spectrogram.unsqueeze(0).to(waveglow.device)
    with torch.no_grad():
        audio = waveglow.infer(mel)
    audio = audio[0].data.cpu().numpy()
    return audio

def save_audio(audio, filename, sample_rate=22050):
    wavfile.write(filename, sample_rate, (audio * 32767).astype('int16'))

if __name__ == "__main__":
    checkpoint_path = f"{MODEL_DIR}/checkpoint_epoch_40.pt"
    waveglow_path = f"{MODEL_DIR}/waveglow_256channels_universal_v5.pt"
    
    model = load_model(checkpoint_path)
    waveglow = load_waveglow(waveglow_path)
    
    text = "This is a test sentence with emotion-based text-to-speech synthesis."
    emotion = "Happy"
    
    mel_outputs, mel_outputs_postnet, alignments = inference(model, text, emotion)
    print("Inference completed.")
    
    # Plot and save mel spectrograms
    plot_mel_spectrogram(mel_outputs[0].cpu().numpy(), "Mel Spectrogram")
    plot_mel_spectrogram(mel_outputs_postnet[0].cpu().numpy(), "Mel Spectrogram (Post-net)")
    
    save_mel_spectrogram(mel_outputs_postnet[0], "mel_spectrogram.npy")
    print("Mel spectrogram saved as 'mel_spectrogram.npy'")
    
    # Generate and save audio
    audio = vocoder_inference(waveglow, mel_outputs_postnet[0])
    save_audio(audio, "output_audio.wav")
    print("Audio saved as 'output_audio.wav'")
    
    print("Text-to-speech synthesis completed successfully!")