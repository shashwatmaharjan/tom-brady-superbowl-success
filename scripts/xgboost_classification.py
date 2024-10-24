import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to perform XGBoost classification for a given year range
def xgboost_classification(train_year_start, train_year_end, features, target, save_directory):
    """
    Train and evaluate an XGBoost classifier on data within a specified range of years.

    Args:
    - train_year_start (int): Start year for training.
    - train_year_end (int): End year for training.
    - features (pd.DataFrame): Feature set for training and testing.
    - target (pd.Series): Target variable (Superbowl Win).
    - save_directory (str): Directory path to save model metrics and plots.

    Returns:
    - model (XGBClassifier): Trained XGBoost model.
    """
    
    # Split the data into training and testing sets
    features_train = features.loc[train_year_start:train_year_end]
    target_train = target.loc[train_year_start:train_year_end]

    features_test = features.loc[train_year_end + 1:]
    target_test = target.loc[train_year_end + 1:]

    # Standardize the features
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    features_test = scaler.transform(features_test)

    # Create the XGBoost model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Fit the model
    model.fit(features_train, target_train)

    # Generate predictions for the test set
    target_pred = model.predict(features_test)

    # Plot and save the confusion matrix
    cm = confusion_matrix(target_test, target_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Superbowl Win', 'Superbowl Win'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{train_year_start}-{train_year_end} prediction for {train_year_end + 1}-2023')
    plt.savefig(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.png'), bbox_inches='tight')

    # Calculate evaluation metrics
    accuracy = accuracy_score(target_test, target_pred)
    precision = precision_score(target_test, target_pred)
    recall = recall_score(target_test, target_pred)
    f1 = f1_score(target_test, target_pred)

    # Save the model metrics to a text file
    with open(os.path.join(save_directory, f'{train_year_start}_{train_year_end}.txt'), 'w') as f:
        f.write(f'Model Metrics for training years {train_year_start} to {train_year_end}:\n')
        f.write(f'Accuracy: {accuracy:.2%}\n')
        f.write(f'Precision: {precision:.2%}\n')
        f.write(f'Recall: {recall:.2%}\n')
        f.write(f'F1 Score: {f1:.2%}\n')
        f.write(f'Confusion Matrix: \n{cm}\n')

    return model

# Main function to orchestrate data loading, processing, and model training
def main():
    # Define directories for data and results
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')
    save_directory = os.path.join(current_directory, 'results', 'classification', 'xgboost')

    # Load the dataset
    data = pd.read_csv(os.path.join(data_directory, 'raw.csv'))

    # Set the index to 'Season' for easier year-based operations
    data = data.set_index('Season')

    # Define feature set and target variable
    features = data.drop('Superbowl Win', axis=1)
    target = data['Superbowl Win']

    # Remove columns with NaN values
    features = features.dropna(axis=1)

    # Train and evaluate the XGBoost model for different year ranges
    model_2002_2009 = xgboost_classification(2002, 2009, features, target, save_directory)
    model_2002_2019 = xgboost_classification(2002, 2019, features, target, save_directory)

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
