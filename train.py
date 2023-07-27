import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from time import time_ns


def etl_pipeline():
    """
    ETL pipeline to process data from CSV to cleaned DataFrame.

    :return: cleaned pandas dataframe
    """
    # Read csv
    loan_data = pd.read_csv('data/loan-approval-data.csv')

    # Remove unnecessary column
    loan_data = loan_data.drop(['Loan_ID'], axis=1)

    # Map Values
    # Label encode gender
    loan_data['Gender'] = loan_data['Gender'].map({'Male': 1, 'Female': 0})

    # label encode education
    loan_data['Education'] = loan_data['Education'].map({'Graduate': 1, 'Not Graduate': 0})

    # Label encode property area
    loan_data['Property_Area'] = loan_data['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})

    # Encode data
    # Label Encoder
    label_encoder = preprocessing.LabelEncoder()

    # Label encode marital status
    loan_data['Married'] = label_encoder.fit_transform(loan_data['Married'])

    # Label encode employer
    loan_data['Self_Employed'] = label_encoder.fit_transform(loan_data['Self_Employed'])

    # Imputation
    # Impute missing dependent values
    loan_data['Dependents'] = loan_data['Dependents'].fillna(loan_data['Dependents'].median())
    loan_data['Dependents'] = loan_data['Dependents'].astype(int)

    # Impute missing loan amount and term values
    loan_data[['LoanAmount', 'Loan_Amount_Term']] = loan_data[['LoanAmount', 'Loan_Amount_Term']].\
        fillna(loan_data[['LoanAmount', 'Loan_Amount_Term']].median())

    # Impute missing credit history
    loan_data['Credit_History'] = loan_data['Credit_History'].fillna(loan_data['Credit_History'].median())
    loan_data['Credit_History'] = loan_data['Credit_History'].astype(int)

    # Change income data type
    loan_data['ApplicantIncome'] = loan_data['ApplicantIncome'].astype(float)

    return loan_data


def train(loan_data):
    """
    Train random forest classifier model on cleaned data.

    :param loan_data: cleaned pandas dataframe
    :return: trained random forest mode, features for test data, labels for test data
    """
    features = loan_data.drop(['Loan_Status'], axis=1)[['ApplicantIncome',
                                                        'CoapplicantIncome',
                                                        'LoanAmount',
                                                        'Credit_History']]
    label = loan_data['Loan_Status']

    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=1)

    rfc_model = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
    rfc_model.fit(x_train, y_train)

    return rfc_model, x_test, y_test


def test(rfc_model, x_test, y_test):
    """
    Test trained random forest classifier model.

    :param rfc_model: random forest classifier model
    :param x_test: features for test data
    :param y_test: labels for test data
    :return: accuracy, precision, and recall metrics
    """
    y_pred = rfc_model.predict(x_test)

    accuracy = "%.3f" % metrics.accuracy_score(y_test, y_pred)
    precision = "%.3f" % metrics.precision_score(y_test, y_pred, pos_label='Y')
    recall = "%.3f" % metrics.recall_score(y_test, y_pred, pos_label='Y')

    print(f'Accuracy: {accuracy} / Precision {precision} / Recall: {recall}')

    return accuracy, precision, recall


if __name__ == '__main__':
    # Start time
    start_time = time_ns()

    # Clean data
    data = etl_pipeline()
    # Train model
    rf_model, test_features, test_labels = train(data)
    # Test Model
    test(rf_model, test_features, test_labels)

    # End time
    end_time = time_ns()

    # Total time
    time_spent = end_time - start_time

    print(f'Time Spent: {time_spent}s')
