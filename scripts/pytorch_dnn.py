import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Define the PyTorch model for deep neural network
class DNNModel(nn.Module):
    """
    A simple feed-forward deep neural network with 3 hidden layers.
    
    Args:
    - input_size (int): Number of input features for the model.
    """
    def __init__(self, input_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
        - x (torch.Tensor): Input features tensor.
        
        Returns:
        - torch.Tensor: Output predictions after applying sigmoid activation.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))
        return x

# Function to train and evaluate the DNN model for a particular time range
def dnn(train_year_start, train_year_end, features, target, save_directory):
    """
    Train and evaluate a deep neural network using data from a specific time period.
    
    Args:
    - train_year_start (int): The start year for the training period.
    - train_year_end (int): The end year for the training period.
    - features (pd.DataFrame): Feature set for training and testing.
    - target (pd.Series): Target variable (Superbowl Win).
    - save_directory (str): Directory path to save model metrics and plots.
    
    Returns:
    - model (DNNModel): Trained DNN model.
    """
    
    # Split data into train and test sets
    features_train = features.loc[train_year_start:train_year_end]
    target_train = target.loc[train_year_start:train_year_end]
    features_test = features.loc[train_year_end + 1:]
    target_test = target.loc[train_year_end + 1:]

    # Standardize the features
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Convert features and target to torch tensors
    features_train = torch.tensor(features_train, dtype=torch.float32)
    target_train = torch.tensor(target_train.values, dtype=torch.float32).view(-1, 1)
    features_test = torch.tensor(features_test, dtype=torch.float32)
    target_test = torch.tensor(target_test.values, dtype=torch.float32).view(-1, 1)

    # Initialize the model, loss function, and optimizer
    model = DNNModel(features_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    for epoch in range(100):  # Train for 100 epochs
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(features_train)  # Forward pass
        loss = criterion(outputs, target_train)  # Compute loss
        loss.backward()  # Backpropagate
        optimizer.step()  # Update weights

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        target_pred = model(features_test)

    # Convert predictions to binary (0 or 1)
    target_pred_binary = (target_pred > 0.5).int()

    # Calculate evaluation metrics
    accuracy = accuracy_score(target_test, target_pred_binary)
    precision = precision_score(target_test, target_pred_binary)
    recall = recall_score(target_test, target_pred_binary)
    f1 = f1_score(target_test, target_pred_binary)

    # Plot and save the confusion matrix
    cm = confusion_matrix(target_test, target_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Superbowl Win', 'Superbowl Win'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{train_year_start}-{train_year_end} prediction for {train_year_end+1}-2023')
    plt.savefig(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.png'), bbox_inches='tight')

    # Save model metrics to a text file
    with open(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.txt'), 'w') as f:
        f.write(f'Model Metrics for training years {train_year_start} to {train_year_end}:\n')
        f.write(f'Accuracy: {accuracy:.2f}\n')
        f.write(f'Precision: {precision:.2f}\n')
        f.write(f'Recall: {recall:.2f}\n')
        f.write(f'F1 Score: {f1:.2f}\n')
        f.write(f'Confusion Matrix: \n{cm}\n')

    return model

# Main function to execute data loading, processing, and model training
def main():
    # Define directories for loading data and saving results
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    save_directory = os.path.join(current_directory, 'results', 'classification', 'pytorch_dnn')

    # Load the dataset
    data = pd.read_csv(os.path.join(data_directory, 'raw.csv'))

    # Set the index to 'Season' for easier year-based operations
    data = data.set_index('Season')

    # Define the feature set and target variable
    features = data.drop('Superbowl Win', axis=1)
    target = data['Superbowl Win']

    # Remove columns with NaN values
    features = features.dropna(axis=1)

    # Train and evaluate the DNN for different year ranges
    model_2002_2009 = dnn(2002, 2009, features, target, save_directory)
    model_2002_2019 = dnn(2002, 2019, features, target, save_directory)

if __name__ == '__main__':
    # Clear the terminal screen for better readability of output
    os.system('cls' if os.name == 'nt' else 'clear')

    # Set default figure size for the plots
    plt.rcParams['figure.figsize'] = (8, 6)

    # Set default font family and math text font for the plots
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Update global font size for plots
    plt.rcParams.update({'font.size': 15})

    main()
