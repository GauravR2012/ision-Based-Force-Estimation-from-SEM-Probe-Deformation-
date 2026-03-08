import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.plot(y_test_actual, label='Actual Force', linestyle='--')
plt.plot(mlp_predictions, label='MLP Predictions')
plt.plot(lstm_predictions, label='LSTM Predictions')
plt.plot(transformer_predictions, label='Transformer Predictions')

plt.xlabel('Sample Index')
plt.ylabel('Force Value')

plt.legend()

plt.title('Actual vs Predicted Force Values')

plt.show()
