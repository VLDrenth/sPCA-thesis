import torch
from torch import nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hyper_params=None, activation=nn.SiLU):
        super(LSTMAutoencoder, self).__init__()

        # Set parameters
        if hyper_params is not None:
            self.hidden_dim = hyper_params.get("hidden_dim", 10)
            self.input_dim = hyper_params.get("input_dim", input_dim)
            self.num_layers = hyper_params.get("num_layers", 1)
            self.activation = hyper_params.get("activation", activation)
            self.gauss_noise = hyper_params.get("gauss_noise", 0)
            self.dropout = hyper_params.get("dropout", 0.0)
        else:
            raise Exception("Hyperparameters not provided")

        # LSTM Encoder
        self.encoder_lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout)

        # Linear layer applied after LSTM encoder
        self.encoder_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # LSTM Decoder
        self.decoder_lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        
        # Linear layer applied after LSTM decoder to reconstruct original dimension
        self.decoder_linear = nn.Linear(self.hidden_dim, input_dim)

        # Initialize weights
        self.init_weights()

        # Define loss criterion
        self.loss_criterion = nn.MSELoss() 

        # Move to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def encode(self, x):
        x = nn.Linear(self.input_dim, 64)(x)
        x, _ = self.encoder_lstm(x)
        x = torch.relu(self.encoder_linear(x))  # Use ReLU activation function after linear layer
        return x
    
    def encode_offline(self, x):
        with torch.no_grad():
            # Ensure x is a tensor
            x = torch.from_numpy(x).float().to(self.device)
            x = x.reshape(x.shape[0], 1, x.shape[1])
            x, _ = self.encoder_lstm(x)

            # Reshape to be 2D
            x = x.reshape(x.shape[0], x.shape[2])

            return x.cpu().numpy()

    def decode(self, x):
        x, _ = self.decoder_lstm(x)
        x = self.decoder_linear(x)  # No activation function is used here, but it may be added if it suits the problem
        return x
    
    def forward(self, input):
        # Add Gaussian noise to input
        input = input + torch.randn(input.shape).to(self.device) * self.gauss_noise

        hidden = self.encode(input)
        output = self.decode(hidden)
        return output
    
    def train(self, X, lr=1e-3, num_epochs=500, batch_size=32):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        num_epochs = num_epochs
        batch_size = batch_size

        # Convert to torch tensor
        X = torch.from_numpy(X).float()

        for epoch in range(num_epochs):
            for i in range(0, X.shape[0], batch_size):
                # Get batch
                batch = X[i:i+batch_size]
                batch = batch.to(self.device)

                batch = batch.reshape(batch.shape[0], 1, batch.shape[1])

                # Forward pass
                output = self(batch)
                loss = self.loss_criterion(output, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Update
                optimizer.step()
            #print(f"Epoch: {epoch}, Loss: {loss.item()}")

