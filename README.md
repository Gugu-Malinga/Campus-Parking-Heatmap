# Campus-Parking-Heatmap
Data-driven analysis of campus parking patterns with visualizations and a machine-learning model for predicting lot fullness by day and hour.

## 📘 UNC Campus Parking Heatmap Analyzer

A data-driven system that analyzes campus parking patterns and predicts lot fullness using machine learning. The project includes visual analytics, feature engineering, and a regression-based prediction model designed to power real-time parking tools.

## 🚀 Project Overview

**Students often struggle with finding parking on campus. This project uses synthetic but realistic parking data from six major lots at UNC Chapel Hill to:**
- Visualize hourly congestion patterns
- Identify peak parking times
- Compare weekday vs weekend availability
- Train a machine-learning model to predict fullness by lot, day, and hour
- Provide clean modular code ready for app integration (Cursor/Streamlit/Flask)

**This project demonstrates skills in:**
-Data cleaning
-Feature engineering
-Visualization
-Regression modeling
-Modular backend architecture

## 📁 Folder Structure
```python
parking_project/
│
├── data/
│   └── unc_parking_synthetic.csv
│
├── scripts/
│   ├── data_loader.py
│   ├── model_training.py
│   └── predictors.py
│
├── app/
│   ├── main.py
│   └── requirements.txt
│
└── README.md
```
## 📊 Dataset Description
**The synthetic dataset contains four weeks of hourly parking observations for:**
- Skipper Bowles Lot
- Craige Deck
- Rams Head Deck
- Stadium Drive Lot
- Bell Tower Lot
- South Road Lot

**Columns include:**
- week
- day_of_week
- hour_24
- lot_name
- fullness_percent
- day_num (engineered)

## 🧠 Core Code Snippets
✔ Load & Clean Data
```python
def load_parking_data(path):
    df = pd.read_csv(path)

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=day_order, ordered=True)

    day_to_num = {day: i for i, day in enumerate(day_order)}
    df["day_num"] = df["day_of_week"].map(day_to_num)

    return df
```
✔ Feature Engineering
```python
def build_features(df):
    X = pd.get_dummies(df[["lot_name", "day_of_week", "hour_24"]], drop_first=False)
    y = df["fullness_percent"]
    return X, y
✔ Model Training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_parking_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
```
✔ Prediction Function
```python
def predict_parking(model, X_columns, lot_name, day_of_week, hour_24):
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    temp = pd.DataFrame({
        "lot_name": [lot_name],
        "day_of_week": pd.Categorical([day_of_week], categories=day_order, ordered=True),
        "hour_24": [hour_24],
    })

    temp_X = pd.get_dummies(temp, columns=["lot_name", "day_of_week"], drop_first=False)
    temp_X = temp_X.reindex(columns=X_columns, fill_value=0)

    return float(model.predict(temp_X)[0])
```
## 📈 Visualizations Included
- Average fullness per lot (bar chart)
- Fullness by hour of day (line chart)
- Weekday vs weekend (bar chart)
- Lot vs hour heatmap using YlOrRd warm color palette

## 🤖 Model Performance
- Using linear regression:
  ~10% Mean Absolute Error (MAE)
- Predicts parking fullness by lot, day, and hour
- This model serves as the basis for a future real-time parking app.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Kaggle Notebooks
(Optional future) Streamlit, Flask, React

## 🔮 Future Improvements
- Deploy a real-time app with student reporting or live sensors
- Add crowdsourcing features

Use Random Forest or XGBoost to improve accuracy

Create a mobile-first UI
