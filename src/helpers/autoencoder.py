import torch.nn as nn
import torch
from functions import get_data

class Autoencoder(nn.Module):

    def __init__(self, hidden_dim=5, input_dim=123, layer_dims = [64, 32, 16], activation=nn.ReLU):
        super(Autoencoder, self).__init__()
        layer_dims = [input_dim] + layer_dims + [hidden_dim]

        self.encoder = create_sequential(layer_dims, activation=activation)
        self.decoder = create_sequential(layer_dims[::-1], activation=activation)
        self.init_weights()
        self.loss_criterion = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, input):
        hidden = self.encoder(input)
        output = self.decoder(hidden)
        return output
    
def create_sequential(layer_dims, activation):
    layers = []
    for i in range(0, len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        if i != len(layer_dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)
    
def train(X, lr, num_epochs=100, batch_size=32):
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = num_epochs
    batch_size = batch_size

    # Convert to torch tensor
    X = torch.from_numpy(X).float()

    for epoch in range(num_epochs):
        for i in range(0, X.shape[0], batch_size):
            # Get batch
            batch = X[i:i+batch_size]
            batch = batch.to(model.device)

            # Forward pass
            output = model(batch)
            loss = model.loss_criterion(output, batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    return model


if __name__ == "__main__":
    X = get_data()['data'].values
    model = train(X, lr=0.0005, num_epochs=100, batch_size=32)


