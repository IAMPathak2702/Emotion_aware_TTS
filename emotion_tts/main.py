import argparse
import os
from emotion_tts.data_preperation.prepare_dataset import prepare_dataset
from emotion_tts.data_preperation.create_metadata import create_metadata
from emotion_tts.model.train import train
from emotion_tts.inference.inference import load_model, inference
from emotion_tts.config import MODEL_DIR, BASE_DIR

def main():
    parser = argparse.ArgumentParser(description="Emotion-aware TTS system")
    parser.add_argument('--prepare', action='store_true', help="Prepare the dataset")
    parser.add_argument('--train', action='store_true', help="Train the model")
    parser.add_argument('--inference', action='store_true', help="Run inference")
    parser.add_argument('--text', type=str, help="Text for inference")
    parser.add_argument('--emotion', type=str, help="Emotion for inference")
    
    args = parser.parse_args()
    
    if args.prepare:
        print("Preparing dataset...")
        prepare_dataset()
        create_metadata()
    
    if args.train:
        print("Training model...")
        train(MODEL_DIR, None, False)
    
    if args.inference:
        if not args.text or not args.emotion:
            print("Please provide --text and --emotion for inference")
            return
        
        print("Running inference...")
        checkpoint_path = os.path.join(MODEL_DIR, "checkpoint_epoch_50.pt")
        model = load_model(checkpoint_path)
        mel_outputs, mel_outputs_postnet, alignments = inference(model, args.text, args.emotion)
        print("Inference completed. Use a vocoder to convert mel spectrograms to audio.")

if __name__ == "__main__":
    main()