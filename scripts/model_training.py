import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def build_features(df):
    """
    Converts categorical columns into one-hot encoded features
    and separates target column from predictors.
    """

    X = pd.get_dummies(df[["lot_name", "day_of_week", "hour_24"]], drop_first=False)
    y = df["fullness_percent"]

    return X, y


def train_parking_model(X, y):
    """
    Splits the dataset and trains a Linear Regression model.
    Returns the trained model.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model
