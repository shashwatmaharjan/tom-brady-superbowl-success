# Import necessary libraries
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Function to perform logistic regresion for particular years
def logistic_regression(train_year_start, train_year_end, features, target):

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

    # Print the model's metrics to percentage and two decimal places
    print(f'Model Metrics for training years {train_year_start} to {train_year_end}:')
    print(f'Accuracy: {accuracy_score(target_test, model.predict(features_test)):.2%}')
    print(f'Precision: {precision_score(target_test, model.predict(features_test)):.2%}')
    print(f'Recall: {recall_score(target_test, model.predict(features_test)):.2%}')
    print(f'F1 Score: {f1_score(target_test, model.predict(features_test)):.2%}')
    print(f'Confusion Matrix: \n{confusion_matrix(target_test, model.predict(features_test))}')
    print()

    return model

# Main function
def main():

    # Define directories
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'data')

    # Load the data
    data = pd.read_csv(os.path.join(data_directory, 'raw.csv'))

    # Replace the indices with the Season
    data = data.set_index('Season')

    # Define the features and target
    features = data.drop('Superbowl Win', axis=1)
    target = data['Superbowl Win']

    # Remove columns with NaN values
    features = features.dropna(axis=1)

    # First use the predictions from year 2002 to 2005 to predict the Superbowl wins from 2006 to 2023
    model = logistic_regression(2002, 2005, features, target)

    # Second use the predictions from year 2002 to 2009 to predict the Superbowl wins from 2010 to 2023
    model = logistic_regression(2002, 2009, features, target)

    # Third use the predictions from year 2002 to 2015 to predict the Superbowl wins from 2016 to 2023
    model = logistic_regression(2002, 2015, features, target)

    # Fourth use the predictions from year 2002 to 2019 to predict the Superbowl wins from 2020 to 2023
    model = logistic_regression(2002, 2019, features, target)

if __name__ == '__main__':

    # Clear the console based on the OS
    os.system('cls' if os.name == 'nt' else 'clear')

    main()