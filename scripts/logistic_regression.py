import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to perform logistic regression for particular years
def logistic_regression(train_year_start, train_year_end, features, target, save_directory):

    features_train = features.loc[train_year_start:train_year_end]
    target_train = target.loc[train_year_start:train_year_end]

    features_test = features.loc[train_year_end+1:]
    target_test = target.loc[train_year_end+1:]

    # Perform standardization on the features
    scaler = StandardScaler()

    # Transform the features
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Create the model
    model = LogisticRegression(max_iter=1000)

    # Fit the model
    model.fit(features_train, target_train)

    # Plot the confusion matrix
    cm = confusion_matrix(target_test, model.predict(features_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Superbowl Win', 'Superbowl Win'])
    disp.plot(cmap=plt.cm.Blues)  # Use a color map for better visuals
    plt.title(f'{train_year_start}-{train_year_end} prediction for {train_year_end+1}-2023')
    plt.savefig(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.png'), bbox_inches='tight')

    # Write the metrics to a file
    with open(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.txt'), 'w') as f:
        f.write(f'Model Metrics for training years {train_year_start} to {train_year_end}:\n')
        f.write(f'Accuracy: {accuracy_score(target_test, model.predict(features_test)):.2%}\n')
        f.write(f'Precision: {precision_score(target_test, model.predict(features_test)):.2%}\n')
        f.write(f'Recall: {recall_score(target_test, model.predict(features_test)):.2%}\n')
        f.write(f'F1 Score: {f1_score(target_test, model.predict(features_test)):.2%}\n')
        f.write(f'Confusion Matrix: \n{confusion_matrix(target_test, model.predict(features_test))}\n')

    return model

# Main function
def main():

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    save_directory = os.path.join(current_directory, 'results', 'classification', 'logistic_regression')

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
    model_2002_2009 = logistic_regression(2002, 2009, features, target, save_directory)

    # Fourth use the predictions from year 2002 to 2019 to predict the Superbowl wins from 2020 to 2023
    model_2002_2019 = logistic_regression(2002, 2019, features, target, save_directory)


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