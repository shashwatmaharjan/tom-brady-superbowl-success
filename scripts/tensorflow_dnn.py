import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import tensorflow as tf

# Function to perform deep neural network for particular years
def dnn(train_year_start, train_year_end, features, target, save_directory):

    features_train = features.loc[train_year_start:train_year_end]
    target_train = target.loc[train_year_start:train_year_end]

    features_test = features.loc[train_year_end+1:]
    target_test = target.loc[train_year_end+1:]

    # Perform standardization on the features
    scaler = StandardScaler()

    # Transform the features
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Create a sequential model
    model = tf.keras.models.Sequential()

    # Add the input layer
    model.add(tf.keras.layers.Input(shape=(features_train.shape[1],)))

    # Add hidden layers
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))

    # Add the output layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

    # Fit the model
    model.fit(features_train, target_train, epochs=100, batch_size=32)
              
    # Predict the target
    target_pred = model.predict(features_test)

    # Convert the predictions to binary
    target_pred_binary = (target_pred > 0.5).astype(int)

    # Calculate the metrics
    accuracy = accuracy_score(target_test, target_pred_binary)
    precision = precision_score(target_test, target_pred_binary)
    recall = recall_score(target_test, target_pred_binary)
    f1 = f1_score(target_test, target_pred_binary)
    
    # Plot the confusion matrix
    cm = confusion_matrix(target_test, target_pred_binary)  # Use binary predictions here
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Superbowl Win', 'Superbowl Win'])
    disp.plot(cmap=plt.cm.Blues)  # Use a color map for better visuals
    plt.title(f'{train_year_start}-{train_year_end} prediction for {train_year_end+1}-2023')
    plt.savefig(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.png'), bbox_inches='tight')

    # Write the metrics to a file with 2 decimal places
    with open(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.txt'), 'w') as f:
        f.write(f'Model Metrics for training years {train_year_start} to {train_year_end}:\n')
        f.write(f'Accuracy: {accuracy:.2f}\n')
        f.write(f'Precision: {precision:.2f}\n')
        f.write(f'Recall: {recall:.2f}\n')
        f.write(f'F1 Score: {f1:.2f}\n')
        f.write(f'Confusion Matrix: \n{confusion_matrix(target_test, target_pred_binary)}\n')

    return model

# Main function
def main():

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    save_directory = os.path.join(current_directory, 'results', 'classification', 'tensorflow_dnn')

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