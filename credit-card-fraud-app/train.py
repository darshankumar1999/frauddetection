import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

def preprocess(df):
    # Encode categorical columns
    cat_cols = ["TransactionID", "TransactionDate", "MerchantID", "TransactionType", "Location"]

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Split features and target
    X = df.drop("IsFraud", axis=1)
    y = df["IsFraud"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerical columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_model(X_train, y_train):
    smote = SMOTE()
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_res, y_train_res)
    return model


def main():
    print("Loading data...")
    df = pd.read_csv("creditcard.csv")

    print("Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("Saving model...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Training complete!")


if __name__ == "__main__":
    main()
