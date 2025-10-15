import os
from datetime import datetime
from dotenv import load_dotenv

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sklearn.linear_model import LogisticRegression


# Load environment variables from .env file if present
load_dotenv()

DB_URI = os.getenv('DB_URI', 'sqlite:///fraud.db')
THRESH_APPROVE = float(os.getenv('THRESH_APPROVE', '0.3'))
THRESH_FLAG = float(os.getenv('THRESH_FLAG', '0.7'))

# Directory paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# SQLAlchemy setup
engine = create_engine(DB_URI, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class Transaction(Base):
    """SQLAlchemy model for the transactions table."""
    __tablename__ = 'transactions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    amount = Column(Float, nullable=False)
    merchant = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    risk_score = Column(Float, nullable=True)
    result = Column(String(20), nullable=True)


# Create tables if they do not exist
Base.metadata.create_all(bind=engine)


def load_global_model():
    """
    Load the global training data and fit a logistic regression model.
    ...
    """
    training_csv = os.path.join(DATASET_DIR, 'training_data.csv')
    df = pd.read_csv(training_csv)

    # --- START: DATA BALANCING LOGIC ---
    fraud_df = df[df['label'] == 1]
    non_fraud_df = df[df['label'] == 0]

    if len(fraud_df) == 0:
        # If there's no fraud data, just use the original df
        balanced_df = df
    else:
        # Oversample the minority class (fraud) to match the majority
        fraud_oversampled = fraud_df.sample(
            n=len(non_fraud_df), 
            replace=True, 
            random_state=42
        )
        # Combine into a new, balanced dataframe
        balanced_df = pd.concat([non_fraud_df, fraud_oversampled])
    
    # Add a print statement for a sanity check
    print(f"--- Training Global Model on {len(balanced_df)} records ({len(non_fraud_df)} non-fraud, {len(balanced_df) - len(non_fraud_df)} fraud) ---")


    # **CRITICAL:** Use the new 'balanced_df' for training
    X = balanced_df[['amount']].values.reshape(-1, 1)
    y = balanced_df['label'].astype(int).values
    
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    return model


# Fit the global model once at startup
GLOBAL_MODEL = load_global_model()


def train_user_model(transactions_df: pd.DataFrame):
    """
    Train a logistic regression model on a user's historical
    transactions.
    ...
    """
    # Fall back to the global model if the user has too few transactions
    # or not enough variety in their history.
    if transactions_df.shape[0] < 10:
        return GLOBAL_MODEL
        
    y_labels = (transactions_df['result'] == 'Fraud').astype(int)
    if len(np.unique(y_labels)) < 2:
        return GLOBAL_MODEL

    # --- START: NEW DATA BALANCING LOGIC FOR USER MODEL ---
    fraud_df = transactions_df[y_labels == 1]
    non_fraud_df = transactions_df[y_labels == 0]

    # Oversample the user's fraud transactions to match non-fraud
    fraud_oversampled = fraud_df.sample(
        n=len(non_fraud_df),
        replace=True,
        random_state=42
    )
    
    balanced_df = pd.concat([non_fraud_df, fraud_oversampled])
    
    print(f"--- Training PERSONALIZED model on {len(balanced_df)} records for this user. ---")

    # **CRITICAL:** Use the new 'balanced_df' for training
    X = balanced_df[['amount']].values.reshape(-1, 1)
    y = (balanced_df['result'] == 'Fraud').astype(int).values
    
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    return model


def classify_risk(probability: float) -> str:

    if probability < THRESH_APPROVE:
        return 'Approved'
    elif probability < THRESH_FLAG:
        return 'Flagged'
    else:
        return 'Fraud'


def create_app() -> Flask:
    """Factory to create and configure the Flask application."""
    app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
    CORS(app) 

    @app.route('/')
    def serve_index():
        """Serve the frontend's index.html file."""
        return send_from_directory(FRONTEND_DIR, 'index.html')

    @app.route('/<path:path>')
    def serve_static(path):
        """Serve static files such as JavaScript and CSS."""
        return send_from_directory(FRONTEND_DIR, path)

    @app.route('/simulate', methods=['POST'])

    def simulate_transaction():

        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "Invalid or empty JSON payload"}), 400

        user_id = payload.get('user_id')
        amount = payload.get('amount')
        merchant = payload.get('merchant')
        timestamp_str = payload.get('timestamp')

        # Validate fields
        try:
            user_id = int(user_id)
            amount = float(amount)
            merchant = str(merchant)
            # Parse timestamp; if none provided, use current time
            if timestamp_str:
                ts = datetime.fromisoformat(timestamp_str)
            else:
                ts = datetime.utcnow()
        except Exception as e:
            return jsonify({"error": f"Invalid payload fields: {e}"}), 400

        session = SessionLocal()
        try:
            # Fetch user's past transactions
            past = session.query(Transaction).filter(Transaction.user_id == user_id).all()
            if past:
                # Convert to DataFrame
                data = pd.DataFrame([
                    {
                        'amount': t.amount,
                        'result': t.result
                    } for t in past
                ])
                model = train_user_model(data)
            else:
                # Use global model if no history
                model = GLOBAL_MODEL

            # Predict probability of fraud using the model
            prob = float(model.predict_proba(np.array([[amount]]))[0][1])
            classification = classify_risk(prob)

            # Persist transaction
            transaction = Transaction(
                user_id=user_id,
                amount=amount,
                merchant=merchant,
                timestamp=ts,
                risk_score=prob,
                result=classification
            )
            session.add(transaction)
            session.commit()

            return jsonify({
                'risk_score': round(prob, 4),
                'result': classification,
                'transaction': {
                    'id': transaction.id,
                    'user_id': user_id,
                    'amount': amount,
                    'merchant': merchant,
                    'timestamp': ts.isoformat(),
                    'risk_score': round(prob, 4),
                    'result': classification
                }
            })
        finally:
            session.close()

    @app.route('/transactions', methods=['GET'])
    def list_transactions():
        """
        Endpoint to fetch all transactions.

        Optional query parameter `user_id` filters transactions for a
        specific user.
        """
        user_id = request.args.get('user_id', type=int)
        session = SessionLocal()
        try:
            query = session.query(Transaction)
            if user_id is not None:
                query = query.filter(Transaction.user_id == user_id)
            records = query.order_by(Transaction.id.desc()).all()
            result = []
            for t in records:
                result.append({
                    'id': t.id,
                    'user_id': t.user_id,
                    'amount': t.amount,
                    'merchant': t.merchant,
                    'timestamp': t.timestamp.isoformat(),
                    'risk_score': t.risk_score,
                    'result': t.result
                })
            return jsonify(result)
        finally:
            session.close()

    return app


if __name__ == '__main__':
    application = create_app()
    # Enable debug mode only for local development
    application.run(host='0.0.0.0', port=5000, debug=True)