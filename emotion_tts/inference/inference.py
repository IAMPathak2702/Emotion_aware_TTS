import torch
from emotion_tts.model.tacotron2_emotion import Tacotron2Emotion
from emotion_tts.config import HPARAMS, EMOTIONS, MODEL_DIR
from emotion_tts.utils.text_processing import text_to_sequence

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

if __name__ == "__main__":
    checkpoint_path = f"{MODEL_DIR}/checkpoint_epoch_50.pt"
    model = load_model(checkpoint_path)
    
    text = "This is a test sentence."
    emotion = "Angry"
    
    mel_outputs, mel_outputs_postnet, alignments = inference(model, text, emotion)
    print("Inference completed. Use a vocoder to convert mel spectrograms to audio.")