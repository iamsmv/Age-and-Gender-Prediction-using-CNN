# Online Payments Fraud Detection System

This project implements a machine learning-based system for detecting fraudulent online payment transactions. The system uses various machine learning algorithms to identify potentially fraudulent transactions in real-time.

## Features

- Data preprocessing and feature engineering
- Multiple model training (Logistic Regression, SVM Linear, SVM RBF)
- Model evaluation with precision-recall curves
- Real-time fraud prediction
- Model persistence and loading
- Comprehensive visualization of results

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Project Structure

- `fraud_detection.py`: Main implementation file containing the FraudDetectionSystem class
- `requirements.txt`: List of required Python packages
- `README.md`: Project documentation

## Usage

1. Prepare your transaction data in CSV format with the following columns:
   - type: Transaction type
   - amount: Transaction amount
   - oldbalanceOrg: Original balance of the sender
   - newbalanceOrig: New balance of the sender
   - oldbalanceDest: Original balance of the recipient
   - newbalanceDest: New balance of the recipient
   - isFraud: Target variable (1 for fraud, 0 for legitimate)

2. Run the main script:
```bash
python fraud_detection.py
```

## Model Training

The system trains multiple models and selects the best performing one based on accuracy. The models include:
- Logistic Regression
- SVM with Linear Kernel
- SVM with RBF Kernel

## Evaluation Metrics

The system provides:
- Classification report with precision, recall, and F1-score
- Confusion matrix visualization
- Precision-Recall curve
- Average precision score

## Real-time Prediction

To use the trained model for real-time prediction:

```python
from fraud_detection import FraudDetectionSystem

# Initialize the system
fraud_system = FraudDetectionSystem()

# Load the trained model
fraud_system.load_model('fraud_detection_model.joblib')

# Prepare new transaction data
new_transaction = pd.DataFrame(...)

# Get predictions
predictions, probabilities = fraud_system.predict_fraud(new_transaction)
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 