import torch.nn as nn
import torch
from src.helpers.functions import get_data


class Encoder(nn.Module):
    
    def __init__(self, layer_dims, activation=nn.ReLU):
        super().__init__()
        
        # Create fully connected layers for encoder
        layers = create_layers(layer_dims, activation)
        
        # Add 1-D convolutional layer to capture temporal dependencies
        layers.insert(0, nn.Conv1d(1, 1, 3, padding=1))
        layers.insert(1, activation())
        layers.insert(2, nn.BatchNorm1d(1))
        #layers.insert(3, nn.Conv1d(1, 1, 3, padding=1))
        #layers.insert(4, activation())
        #layers.insert(5, nn.BatchNorm1d(1))

        # Create sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        output = self.model(input)
        return output


class Decoder(nn.Module):
    
    def __init__(self, layer_dims, activation=nn.ReLU):
        super().__init__()

        # Fully connected part is the same as encoder but in reverse
        layer_dims = layer_dims[::-1]
        layers = create_layers(layer_dims, activation)

        # Adds 1-D convolutional layer at the end to capture temporal dependencies
        
        layers.append(nn.ConvTranspose1d(1, 1, 3, padding=1))
        layers.append(activation())
        layers.append(nn.BatchNorm1d(1))
        #layers.append(nn.ConvTranspose1d(1, 1, 3, padding=1))
        #layers.append(activation())
        #layers.append(nn.BatchNorm1d(1))

        # Create sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        output = self.model(input)

        return output

class Autoencoder(nn.Module):

    def __init__(self, input_dim=123, activation=nn.SiLU, hyper_params=None):
        super(Autoencoder, self).__init__()
        # Initialize hyperparameters
        if hyper_params is not None:
            self.hidden_dim = hyper_params.get("hidden_dim", 10)
            self.input_dim = hyper_params.get("input_dim", input_dim)
            self.layer_dims = hyper_params.get("layer_dims", [64])
            self.activation = hyper_params.get("activation", activation)
            self.gauss_noise = hyper_params.get("gauss_noise", 0)
            self.dropout = hyper_params.get("dropout", 0.0)
        else:
            raise Exception("Hyperparameters not provided")

        self.layer_dims = [input_dim] + self.layer_dims + [self.hidden_dim]

        self.encoder = Encoder(self.layer_dims, activation=activation)
        self.decoder = Decoder(self.layer_dims, activation=activation)

        self.init_weights()

        self.loss_criterion = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, input):
        # Add Gaussian noise to input
        input = input + torch.randn(input.shape).to(self.device) * self.gauss_noise

        hidden = self.encoder(input)
        output = self.decoder(hidden)
        return output
    
    def encode(self, input):
        self.eval()
        input = input.reshape(input.shape[0], 1, input.shape[1])

        # Not training the model so no need to compute gradients
        with torch.no_grad():
            # Ensure input is tensor and not numpy array
            input = torch.from_numpy(input).float()
            output = self.encoder(input)

            # Resahpe output to be 2D
            output = output.reshape(output.shape[0], output.shape[2])
            
            # Return numpy array
            return output.cpu().detach().numpy()
        
    def train_model(self, X, lr=1e-3, num_epochs=100, batch_size=64):
        self.train()
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
            #if epoch % 100 == 0:
            #print(f"Epoch: {epoch}, Loss: {loss.item()}")

    
def create_layers(layer_dims, activation, dropout=0.2):
    layers = []
    for i in range(0, len(layer_dims) - 1):
        # Linear layer
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        if i != len(layer_dims) - 2:

            # Non-linearity
            layers.append(activation())

            # Dropout
            layers.append(nn.Dropout(dropout))
       
    return layers
    
if __name__ == "__main__":
    X = get_data()['data'].values
    X_train = X[:360]
    X_test = X[360:]
    model = Autoencoder(hidden_dim=10, input_dim=123, layer_dims = [64, 32, 16], activation=nn.ReLU)
    model.train_model(X_train, lr=0.0005, num_epochs=200, batch_size=32)

    # Get the latent representation of the data for the test set
    X_test = torch.from_numpy(X_test).float()
    X_test = X_test.to(model.device)
    latent = model.encode(X_test).cpu().detach().numpy()