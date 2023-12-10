from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter

epochs = 1000

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter('/content/tensorboard')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the CRNN model
crnn_model.to(device)

# Define the loss function and optimizer
criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
optimizer = optim.Adam(crnn_model.parameters(), betas=(0.5,0.999))
average_loss = []

# Training loop
for epoch in range(epochs):
    crnn_model.train()  # Set the model to training mode
    total_loss = 0
    optimizer.zero_grad() # Zero out gradient for each epoch of training
    for i, (images, transcriptions) in enumerate(data_loader):
        
        images = images.to(device)
        transcriptions = transcriptions.to(device)
        transcription_lengths = [len(t) for t in transcriptions]

        outputs = crnn_model(images)
        input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)
        target_lengths = torch.randint(low=0, high=100, size=(images.size(0),), dtype=torch.long).to(device)
      
        loss = criterion(outputs, transcriptions, input_lengths, target_lengths)
        total_loss += loss.item()

        # Backward and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(crnn_model.parameters(), max_norm=1.0)
        optimizer.step()

        # Log to TensorBoard
        writer.add_scalar('Training Loss', loss.item(), epoch * len(data_loader) + i)

    # Calculate average loss for the epoch
    average_loss.append(total_loss / (i + 1))

    # Print average loss for the epoch
    print(f'Epoch {1+epoch:2d}/{epochs}, Train Loss: {total_loss / (i + 1):.6f}')

    # Log average loss to TensorBoard
    writer.add_scalar('Average Training Loss', total_loss / (i + 1), epoch)

    # Save checkpoint
    torch.save(crnn_model.state_dict(), f'/content/crnn_epoch_{epoch}.pth')

# Close the writer when we are done
writer.close()

# After the training loop
plt.plot(range(epochs), average_loss, linestyle='solid', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

