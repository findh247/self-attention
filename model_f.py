import torch

def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

#????????????????
    mels, labels = batch

    mels = mels.to(device)
    labels = labels.to(device)
    outs = model(mels)
    loss = criterion(outs, labels)
    # Get the speaker id with highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy