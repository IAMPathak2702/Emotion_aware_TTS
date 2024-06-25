import torch
from torch.utils.data import DataLoader
from emotion_tts.model.tacotron2_emotion import Tacotron2Emotion
from emotion_tts.config import HPARAMS, MODEL_DIR
from emotion_tts.utils.data_utils import TextMelLoader, TextMelCollate

class Tacotron2Loss(torch.nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = torch.nn.MSELoss()(mel_out, mel_target) + \
            torch.nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = torch.nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss

def train(output_directory, checkpoint_path, warm_start):
    hparams = HPARAMS
    model = Tacotron2Emotion(hparams)
    
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    criterion = Tacotron2Loss()
    
    train_loader = DataLoader(
        TextMelLoader(hparams['training_files'], hparams),
        batch_size=hparams['batch_size'], shuffle=True,
        collate_fn=TextMelCollate(1))
    
    val_loader = DataLoader(
        TextMelLoader(hparams['val_files'], hparams),
        batch_size=hparams['batch_size'], shuffle=False,
        collate_fn=TextMelCollate(1))
    
    for epoch in range(hparams['epochs']):
        model.train()
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, batch in enumerate(val_loader):
                x, y = model.parse_batch(batch)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
            val_loss /= (i + 1)
            print(f"Epoch: {epoch}, Validation Loss: {val_loss}")
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{MODEL_DIR}/checkpoint_epoch_{epoch}.pt")

if __name__ == "__main__":
    train(MODEL_DIR, None, False)