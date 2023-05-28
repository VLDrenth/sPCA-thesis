import torch.nn as nn
import torch
from src.helpers.functions import get_data

class Autoencoder(nn.Module):

    def __init__(self, hidden_dim=10, input_dim=123, layer_dims = [64, 32, 16], activation=nn.ReLU):
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
    
    def encode(self, input):
        # Not training the model so no need to compute gradients
        with torch.no_grad():
            # Ensure input is tensor and not numpy array
            input = torch.from_numpy(input).float()
            output = self.encoder(input)
            
            # Return numpy array
            return output.cpu().detach().numpy()
        
    def train(self, X, lr=1e-3, num_epochs=100, batch_size=32):
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

                # Forward pass
                output = self(batch)
                loss = self.loss_criterion(output, batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Update
                optimizer.step()
            #print(f"Epoch: {epoch}, Loss: {loss.item()}")

    
def create_sequential(layer_dims, activation):
    layers = []
    for i in range(0, len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        if i != len(layer_dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)
    


if __name__ == "__main__":
    X = get_data()['data'].values
    X_train = X[:360]
    X_test = X[360:]
    model = Autoencoder(hidden_dim=10, input_dim=123, layer_dims = [64, 32, 16], activation=nn.ReLU)
    model.train(X_train, lr=0.0005, num_epochs=200, batch_size=32)

    # Get the latent representation of the data for the test set
    X_test = torch.from_numpy(X_test).float()
    X_test = X_test.to(model.device)
    latent = model.encode(X_test).cpu().detach().numpy()





