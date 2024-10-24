import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Define the PyTorch model
class DNNModel(nn.Module):
    def __init__(self, input_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))
        return x

# Function to perform logistic regression for particular years
def dnn(train_year_start, train_year_end, features, target, save_directory):
    
    # Split data into train and test sets based on the year
    features_train = features.loc[train_year_start:train_year_end]
    target_train = target.loc[train_year_start:train_year_end]

    features_test = features.loc[train_year_end+1:]
    target_test = target.loc[train_year_end+1:]

    # Perform standardization on the features
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Convert features and target to torch tensors
    features_train = torch.tensor(features_train, dtype=torch.float32)
    target_train = torch.tensor(target_train.values, dtype=torch.float32).view(-1, 1)
    
    features_test = torch.tensor(features_test, dtype=torch.float32)
    target_test = torch.tensor(target_test.values, dtype=torch.float32).view(-1, 1)

    # Define model, loss, and optimizer
    model = DNNModel(features_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    model.train()
    for epoch in range(100):  # Train for 100 epochs
        optimizer.zero_grad()
        outputs = model(features_train)
        loss = criterion(outputs, target_train)
        loss.backward()
        optimizer.step()

    # Testing the model
    model.eval()
    with torch.no_grad():
        target_pred = model(features_test)

    # Convert predictions to binary
    target_pred_binary = (target_pred > 0.5).int()

    # Calculate the metrics
    accuracy = accuracy_score(target_test, target_pred_binary)
    precision = precision_score(target_test, target_pred_binary)
    recall = recall_score(target_test, target_pred_binary)
    f1 = f1_score(target_test, target_pred_binary)

    # Plot the confusion matrix
    cm = confusion_matrix(target_test, target_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Superbowl Win', 'Superbowl Win'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{train_year_start}-{train_year_end} prediction for {train_year_end+1}-2023')
    plt.savefig(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.png'), bbox_inches='tight')

    # Write the metrics to a file
    with open(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.txt'), 'w') as f:
        f.write(f'Model Metrics for training years {train_year_start} to {train_year_end}:\n')
        f.write(f'Accuracy: {accuracy:.2f}\n')
        f.write(f'Precision: {precision:.2f}\n')
        f.write(f'Recall: {recall:.2f}\n')
        f.write(f'F1 Score: {f1:.2f}\n')
        f.write(f'Confusion Matrix: \n{cm}\n')

    return model

# Main function
def main():
    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    save_directory = os.path.join(current_directory, 'results', 'classification', 'pytorch_dnn')

    # Load the data
    data = pd.read_csv(os.path.join(data_directory, 'raw.csv'))

    # Replace the indices with the Season
    data = data.set_index('Season')

    # Define the features and target
    features = data.drop('Superbowl Win', axis=1)
    target = data['Superbowl Win']

    # Remove columns with NaN values
    features = features.dropna(axis=1)

    # Second use the predictions from year 2002 to 2009 to predict the Superbowl wins from 2010 to 2023
    model_2002_2009 = dnn(2002, 2009, features, target, save_directory)

    # Fourth use the predictions from year 2002 to 2019 to predict the Superbowl wins from 2020 to 2023
    model_2002_2019 = dnn(2002, 2019, features, target, save_directory)

if __name__ == '__main__':
    # Clear the terminal screen for better readability of output
    os.system('cls' if os.name == 'nt' else 'clear')

    # Set default figure size for the plots
    plt.rcParams['figure.figsize'] = (8, 6)

    # Set default font family and math text font for the plots
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Update font size globally for plots
    plt.rcParams.update({'font.size': 15})

    main()
