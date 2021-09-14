import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report


def clean_dataset(df):
    '''
    The purpose of the function clean_dataset_for_mining is to clean and prepare the data set for analysis.
    :param df: an original dataframe
    :return: a cleaned dataframe ready for visualization
    '''
    # Drop useless features
    df = df.iloc[:, [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 16, 17]]

    # Drop levels with little records
    df = df[~df['PERP_AGE_GROUP'].isin(["UNKNOWN", '1020', '940', '224'])]
    df = df[df['VIC_AGE_GROUP'] != "UNKNOWN"]
    df = df[~df['PERP_RACE'].isin(['ASIAN / PACIFIC ISLANDER', 'AMERICAN INDIAN/ALASKAN NATIVE'])]
    df = df[~df['VIC_RACE'].isin(['UNKNOWN', 'AMERICAN INDIAN/ALASKAN NATIVE'])]

    # Since elder groups are too small, we combine them together into a bigger group
    new_class = {'18-24': '18-24', '25-44': '25-44', '<18': '<18', '45-64': '45+', '65+': '45+'}
    df['PERP_AGE_GROUP'] = df['PERP_AGE_GROUP'].map(new_class)
    df['VIC_AGE_GROUP'] = df['VIC_AGE_GROUP'].map(new_class)

    # Convert datetime to month, year, and day of the week
    df['OCCUR_DATE'] = pd.to_datetime(df['OCCUR_DATE'])

    df['month'] = df['OCCUR_DATE'].dt.month
    df['year'] = df['OCCUR_DATE'].dt.year
    df['weekday'] = df['OCCUR_DATE'].dt.day_name()

    # Bin 'OCCUR_time' into 4 time sessions
    df['OCCUR_TIME'] = pd.to_datetime(df['OCCUR_TIME'], format='%H:%M:%S')

    bins = [0, 6, 12, 18, 23]
    labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['OCCUR_TIME'] = pd.cut(df['OCCUR_TIME'].dt.hour, bins=bins, labels=labels, include_lowest=True)

    return df


def month_plotting(df):
    # Exploring month
    # Group by month
    month_trend = df.groupby(['year', 'month']).count().reset_index().iloc[:, 0:3].rename(
        columns={'OCCUR_DATE': 'Shooting Cases'})

    # Plot month trend by year
    month_plot = px.bar(month_trend, x="month", y="Shooting Cases", color='year', barmode='group')
    month_plot.update_layout(title='Monthly Shooting Incidents by Year',
                             xaxis_title='Month',
                             yaxis_title='Total Incidents')

    month_plot.show()


def time_weekday_plotting(df):
    # Exploring time period and weekday frequency
    # Group by time period(Only focus on recent 3 years)
    tmp = df[df['year'].isin([2019, 2020, 2021])]
    time_trend = tmp.groupby('OCCUR_TIME').count().reset_index().iloc[:, 0:2].rename(
        columns={'OCCUR_DATE': 'Shooting Cases'})

    # Group by weekday(Only focus on recent 3 years)
    weekday_trend = tmp.groupby('weekday').count().reset_index().iloc[:, 0:2].rename(
        columns={'OCCUR_DATE': 'Shooting Cases'})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Shooting Frequency in the Recent 3 Years')

    sns.barplot(ax=ax1, x="OCCUR_TIME", y="Shooting Cases", data=time_trend)
    sns.barplot(ax=ax2, x="weekday", y="Shooting Cases", data=weekday_trend)

    ax1.set(xlabel='Time Period', ylabel='Total Incidents', title='By Time Period')
    ax2.set(xlabel='Weekday', ylabel='Total Incidents', title='By Weekday')

    fig.subplots_adjust(wspace=0.2)

    plt.show()


def encode_split_dataset(df, test_size=0.3, rand_state=42):
    """
    The purpose of the function split_dataset is to conduct feature engineering and split the dataset into training and testing sets.
    :param df: the dataset to be split
    :param test_size: the ratio of splitting
    :param rand_state: a seed to ensure the same splits at each generation
    :return: four dataset in the order of X_train, X_test, y_train, y_test
    """
    # Drop missing values on cases without perpetuity information
    df = df.dropna(subset=['PERP_AGE_GROUP', 'PERP_SEX', 'PERP_RACE'])

    # Split into explanatory and response variables
    X = df.drop(['PERP_AGE_GROUP', 'OCCUR_DATE', 'year'], axis=1)
    y = df['PERP_AGE_GROUP']

    # Encode categorical variables
    X = pd.get_dummies(X, columns=X.columns)

    # Standardize features
    X = StandardScaler().fit_transform(X)

    # Split into train and test dataset
    # 'stratify' ensure the same distribution of classes across the train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state,
                                                        stratify=df['PERP_AGE_GROUP'])

    return X_train, X_test, y_train, y_test


def class_distribution(y_train, y_test):
    # Distribution of class attribute in test set
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(top=0.8, wspace=0.3)

    ax1.bar(y_train.value_counts().index, y_train.value_counts())
    ax1.set_title('Train Set')
    ax2.bar(y_test.value_counts().index, y_test.value_counts())
    ax2.set_title('Test Set')

    fig.suptitle('Class Distribution of Train and Test Set')

    plt.show()


def initial_train_evaluate(df, clfs):
    # Split dataset
    X_train, X_test, y_train, y_test = encode_split_dataset(df)

    # Iterate through each classifier
    for name, CLF in clfs.items():
        # Instantiate model
        clf = CLF

        # Train model
        clf.fit(X_train, y_train)

        # Predict on test set
        y_pred = clf.predict(X_test)

        # Evaluation report
        print(f'Initial model parameters: {clf}\n')

        print(f'Accuracy of {name} Tuned model: {accuracy_score(y_test, y_pred)}\n')

        print(f'Classification Report of {name} Initial model:')
        print(classification_report(y_test, y_pred))

        print(f'Confusion Matrix of {name} Initial model:')
        print(confusion_matrix(y_test, y_pred))

        print('\n---------------------------------------------------------------------------------------------')


def tune_models(df, params):
    # Split dataset
    X_train, X_test, y_train, y_test = encode_split_dataset(df)

    # Iterate through each classifier
    for name, [clf, param_grid] in params.items():
        # Tune parameters using GridSearchCV
        tune_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3)
        tune_clf.fit(X_train, y_train)

        print(f'Best parameters of {name} found:\n {tune_clf.best_params_}\n')

        # Train new models
        new_clf = clf.set_params(**tune_clf.best_params_)
        new_clf.fit(X_train, y_train)

        # Predict on test set
        y_pred = new_clf.predict(X_test)

        # Evaluation report
        print(f'Tuned model parameters: {new_clf}\n')

        print(f'Accuracy of {name} Tuned model: {accuracy_score(y_test, y_pred)}\n')

        print(f'Classification Report of {name} Tuned model:')
        print(classification_report(y_test, y_pred))

        print(f'Confusion Matrix of {name} Tuned model:')
        print(confusion_matrix(y_test, y_pred))

        print('\n---------------------------------------------------------------------------------------------')


def main():
    df_ori = pd.read_csv('NYPD_Shooting_Incident_Data__Historic_.csv')
    df = clean_dataset(df_ori)
    df.head()


if __name__ == '__main__':
    main()
