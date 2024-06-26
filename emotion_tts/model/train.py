import torch
from torch.utils.data import DataLoader
from emotion_tts.model.tacotron2_emotion import Tacotron2Emotion
from emotion_tts.config import HPARAMS, MODEL_DIR
from emotion_tts.utils.data_utils import TextMelLoader, TextMelCollate
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    model = Tacotron2Emotion(hparams).to(device)
    model = model.to(device)  # Ensure all model parameters are on the correct device

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    learning_rate = hparams['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = Tacotron2Loss().to(device)

    train_loader = DataLoader(
        TextMelLoader(hparams['training_files'], hparams),
        batch_size=hparams['batch_size'], shuffle=True,
        collate_fn=TextMelCollate(1), pin_memory=True, num_workers=4
    )

    val_loader = DataLoader(
        TextMelLoader(hparams['val_files'], hparams),
        batch_size=hparams['batch_size'], shuffle=False,
        collate_fn=TextMelCollate(1), pin_memory=True, num_workers=4
    )

    os.makedirs(output_directory, exist_ok=True)

    for epoch in range(hparams['epochs']):
        model.train()
        for i, batch in enumerate(train_loader):
            model.zero_grad()
            x, y = model.parse_batch(batch)
            
            # Move all tensors in x to the device, except for text_lengths
            x = list(x)  # Convert tuple to list for modification
            for j in range(len(x)):
                if isinstance(x[j], torch.Tensor):
                    if j == 1:  # Assuming index 1 is text_lengths
                        x[j] = x[j].cpu()  # Keep text_lengths on CPU
                    else:
                        x[j] = x[j].to(device)
            x = tuple(x)  # Convert back to tuple
            
            y = tuple(elem.to(device) for elem in y)
            
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
                x = {k: v.to(device) for k, v in x.items()}
                y = tuple(elem.to(device) for elem in y)
                
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
            val_loss /= (i + 1)
            print(f"Epoch: {epoch}, Validation Loss: {val_loss}")

        if epoch % 7 == 0:
            checkpoint_path = os.path.join(output_directory, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

if __name__ == "__main__":
    train(MODEL_DIR, None, False)