import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to perform logistic regression for a specified range of years
def logistic_regression(train_year_start, train_year_end, features, target, save_directory):
    """
    Train a logistic regression model using the data from the given range of years 
    and evaluate it on the years that follow the training period.
    
    Parameters:
    - train_year_start: Start year of the training period.
    - train_year_end: End year of the training period.
    - features: DataFrame containing feature variables.
    - target: Series containing the target variable (Superbowl Win).
    - save_directory: Path to the directory where outputs (plots and metrics) are saved.
    
    Returns:
    - model: Trained Logistic Regression model.
    """
    
    # Split the data into training and testing sets
    features_train = features.loc[train_year_start:train_year_end]
    target_train = target.loc[train_year_start:train_year_end]
    features_test = features.loc[train_year_end+1:]
    target_test = target.loc[train_year_end+1:]

    # Standardize the features
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Initialize and fit the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(features_train, target_train)

    # Generate and display the confusion matrix
    cm = confusion_matrix(target_test, model.predict(features_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Superbowl Win', 'Superbowl Win'])
    disp.plot(cmap=plt.cm.Blues)  # Use a blue color map for enhanced readability
    plt.title(f'{train_year_start}-{train_year_end} prediction for {train_year_end+1}-2023')
    
    # Save the confusion matrix plot
    plt.savefig(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.png'), bbox_inches='tight')

    # Save the model metrics to a text file
    with open(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.txt'), 'w') as f:
        f.write(f'Model Metrics for training years {train_year_start} to {train_year_end}:\n')
        f.write(f'Accuracy: {accuracy_score(target_test, model.predict(features_test)):.2%}\n')
        f.write(f'Precision: {precision_score(target_test, model.predict(features_test)):.2%}\n')
        f.write(f'Recall: {recall_score(target_test, model.predict(features_test)):.2%}\n')
        f.write(f'F1 Score: {f1_score(target_test, model.predict(features_test)):.2%}\n')
        f.write(f'Confusion Matrix: \n{cm}\n')

    return model

# Main function to orchestrate the data loading, processing, and model training
def main():
    # Define paths for the data and results
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    save_directory = os.path.join(current_directory, 'results', 'classification', 'logistic_regression')

    # Load the data into a DataFrame
    data = pd.read_csv(os.path.join(data_directory, 'raw.csv'))

    # Set the index of the DataFrame to the 'Season' column
    data = data.set_index('Season')

    # Separate the feature set from the target variable
    features = data.drop('Superbowl Win', axis=1)
    target = data['Superbowl Win']

    # Drop columns containing NaN values from the features
    features = features.dropna(axis=1)

    # Train and test models for the specified time periods
    model_2002_2009 = logistic_regression(2002, 2009, features, target, save_directory)
    model_2002_2019 = logistic_regression(2002, 2019, features, target, save_directory)

if __name__ == '__main__':
    # Clear the terminal screen for better output visibility
    os.system('cls' if os.name == 'nt' else 'clear')

    # Set global plotting parameters for visual consistency
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams.update({'font.size': 15})

    main()
