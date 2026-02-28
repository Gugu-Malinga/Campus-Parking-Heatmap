import pandas as pd

def load_parking_data(path):
    """
    Loads and prepares the UNC parking dataset.

    - Reads the CSV file from the given path
    - Ensures the weekday order is correct
    - Creates a numeric day column for modeling
    """

    # Load CSV
    df = pd.read_csv(path)

    # Ensure Monday → Sunday ordering
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=day_order, ordered=True)

    # Convert day names to numeric (Mon=0 ... Sun=6)
    day_to_num = {day: i for i, day in enumerate(day_order)}
    df["day_num"] = df["day_of_week"].map(day_to_num)

    return df
