import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size, num_rnn_layers):
        super(CRNN, self).__init__()
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.num_rnn_layers = num_rnn_layers

        self.conv_layers = nn.Sequential(
          nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 193, 26)

          nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 96, 13)

          nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (256, 48, 6)

          nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (512, 24, 3)

          nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1)),  # Output: (512, 1, 3)
        )

        # Map to sequence layer
        self.map_to_seq = nn.Linear(512 * 24, rnn_hidden_size)

        # Recurrent layers
        self.rnn = nn.LSTM(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            bidirectional=True
        )

        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)  # Times 2 for bidirectional
    
    def forward(self, x):
        # Apply convolutional layers
        conv = self.conv_layers(x)
        # print("After Conv Layers:", conv.size())

        # Ensure the height after convolutions is 1
        assert conv.size(2) == 1, "Height of conv must be 1"

        # Prepare for RNN layers
        conv = conv.squeeze(2)  # Remove the height dimension
        conv = conv.permute(2, 0, 1)  # Change to (w, b, c)

        # Reshape tensor before linear layer
        batch_size = x.size(0)
        conv = conv.squeeze(2)  # Remove the height dimension
        conv = conv.permute(2, 0, 1)  # Change to (w, b, c)

        # Flatten the tensor while keeping the batch dimension
        seq = self.map_to_seq(conv.reshape(batch_size, -1))

        # RNN layers
        recurrent, _ = self.rnn(seq.view(-1, batch_size, self.rnn_hidden_size))

        # Fully connected layer
        T, b, h = recurrent.size()  # sequence length, batch size, hidden dimensions
        t_rec = recurrent.view(T * b, h)  # Reshape for fully connected layer
        output = self.fc(t_rec)

        # Reshape back to (sequence length, batch, num_classes)
        output = output.view(T, b, -1)

        # Apply log_softmax
        output = F.log_softmax(output, dim=2)

        return output




# Define the character set that the OCR needs to recognize
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;'\"-!@*&%$()+=_-[]{}\|/?<>:\#$"
characters += ' '  # Adding a space for CTC blank token

# Instantiate the model
num_classes = len(characters)  # Including the CTC blank character
rnn_hidden_size = 256
num_rnn_layers = 2
crnn_model = CRNN(num_classes, rnn_hidden_size, num_rnn_layers)
