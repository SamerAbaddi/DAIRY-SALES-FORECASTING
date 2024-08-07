import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
import matplotlib.pyplot as plt

# Manually create the dataset from the given data
data = {
    "Week": list(range(1, 44)),
    "Net Sales": [158, 97, 111, 33, 86, 74, 43, 125, 90, 102, 80, 64, 62, 74, 75, 82, 80, 75, 70, 90, 71, 14, 68, 39, 72, 64, 74, 88, 88, 126, 84, 67, 84, 88, 44, 96, 98, 82, 65, 30, 74, 69, 61]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Use the first 30 weeks for training
train_data = df[df['Week'] <= 30]['Net Sales'].values.reshape(-1, 1)
test_data = df[df['Week'] > 30]['Net Sales'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Create training sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # using a longer sequence length
X_train, y_train = create_sequences(train_data_scaled, seq_length)
X_test, y_test = create_sequences(test_data_scaled, seq_length)

# Build the neural network model
model = Sequential()
model.add(Input(shape=(seq_length, 1)))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model and record the training history
history = model.fit(X_train, y_train, epochs=1000, verbose=0, validation_data=(X_test, y_test))

# Forecasting
predictions = []
current_seq = train_data_scaled[-seq_length:]

for i in range(len(test_data)):
    pred = model.predict(current_seq.reshape(1, seq_length, 1))
    predictions.append(pred[0][0])
    current_seq = np.append(current_seq[1:], pred, axis=0)

# Inverse transform the predictions
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Prepare the results for saving
forecast_weeks = list(range(31, 44))
forecast_results = pd.DataFrame({"Week": forecast_weeks, "Forecasted Net Sales": predictions.flatten()})

# Save the results to a CSV file
forecast_results.to_csv("forecast_results.csv", index=False)

# Forecast from Week 44 to Week 80 with added noise (Extended)
extended_forecast_weeks = list(range(44, 81))
extended_predictions = []
noise_factor = 0.1  # Adjust the noise factor as needed

for i in range(len(extended_forecast_weeks)):
    pred = model.predict(current_seq.reshape(1, seq_length, 1))
    noise = np.random.normal(0, noise_factor) * pred[0][0]
    pred_with_noise = pred[0][0] + noise
    extended_predictions.append(pred_with_noise)
    current_seq = np.append(current_seq[1:], pred, axis=0)

# Inverse transform the extended predictions
extended_predictions = scaler.inverse_transform(np.array(extended_predictions).reshape(-1, 1))

# Prepare the extended forecast results for saving
extended_forecast_results = pd.DataFrame({"Week": extended_forecast_weeks, "Forecasted Net Sales": extended_predictions.flatten()})

# Save the extended results to a CSV file
extended_forecast_results.to_csv("extended_forecast_results.csv", index=False)

# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(df['Week'], df['Net Sales'], label='Historical Data', color='blue')
plt.plot(forecast_results['Week'], forecast_results['Forecasted Net Sales'], label='Forecasted Data', color='red')
plt.plot(extended_forecast_results['Week'], extended_forecast_results['Forecasted Net Sales'], label='Extended Forecasted Data', color='green')
plt.axvline(x=43, linestyle='--', color='gray', label='New Forecast Start')
plt.xlabel('Week')
plt.ylabel('Net Sales')
plt.title('Net Sales Forecast')
plt.legend()
plt.show()

# Plotting the training loss curve
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plotting prediction vs actual values
plt.figure(figsize=(12, 6))
plt.plot(range(len(test_data)), scaler.inverse_transform(y_test), label='Actual Net Sales', color='blue')
plt.plot(range(len(test_data)), predictions, label='Predicted Net Sales', color='red')
plt.xlabel('Time Step')
plt.ylabel('Net Sales')
plt.title('Actual vs Predicted Net Sales')
plt.legend()
plt.show()

# Plotting residuals
residuals = scaler.inverse_transform(y_test) - predictions
plt.figure(figsize=(12, 6))
plt.plot(range(len(residuals)), residuals, label='Residuals', color='blue')
plt.axhline(y=0, linestyle='--', color='gray')
plt.xlabel('Time Step')
plt.ylabel('Residual')
plt.title('Residuals of Predictions')
plt.legend()
plt.show()

print("Forecasting completed. Results saved to 'forecast_results.csv' and 'extended_forecast_results.csv'.")
