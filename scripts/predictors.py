def predict_parking(model, X_columns, lot_name, day_of_week, hour_24):
    """
    Produces a parking fullness prediction for a given
    lot, day, and hour using the trained model.
    """

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Build input row
    temp = pd.DataFrame({
        "lot_name": [lot_name],
        "day_of_week": pd.Categorical([day_of_week], categories=day_order, ordered=True),
        "hour_24": [hour_24],
    })

    # One-hot encode input
    temp_X = pd.get_dummies(temp, columns=["lot_name", "day_of_week"], drop_first=False)

    # Align columns to the model's training feature names
    temp_X = temp_X.reindex(columns=X_columns, fill_value=0)

    # Predict
    return float(model.predict(temp_X)[0])
