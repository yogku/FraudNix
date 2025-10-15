

# FraudNix

**GitHub Repository:** [https://github.com/yogku/FraudNix](https://github.com/yogku/FraudNix)

This project is a complete full‑stack demo for simulating financial transactions and classifying them as **Approved**, **Flagged** or **Fraudulent** using a live‑trained machine learning model. The system consists of a Python/Flask backend, a relational database for persistence, a simple HTML/JavaScript frontend and a small data preparation pipeline.

-----

## Features

  * **Responsive web interface** with a dropdown to select a user and a “Simulate Transaction” button.
  * Each click generates a random transaction and sends it to the backend for processing.
  * Past transactions are displayed in a real‑time table with the transaction details, risk score and classification.
  * A summary chart shows the counts of Approved, Flagged and Fraud transactions for the currently selected user.
  * The backend uses **scikit‑learn** to train or update a logistic regression model on each user’s transaction history. If a user has insufficient history, a global model trained on a pre‑generated dataset is used.
  * All transactions and their classifications are persisted in a database (SQLite by default; MySQL can be used by changing a single environment variable).

-----

## Repository Structure

```
fraud_detection_app/
├── backend/
│   ├── app.py        # Flask application with REST API
│   ├── seed_db.py    # Script to populate the transactions table
│   └── __init__.py   # (empty) makes backend a Python package
├── dataset/
│   ├── training_data.csv
│   ├── test_data.csv
│   └── simulation_data.csv
├── frontend/
│   ├── index.html    # Main web page
│   ├── script.js     # Frontend logic (fetches API, updates UI)
│   └── styles.css    # Minimal styling
├── prepare_data.py   # Generates the synthetic dataset
├── requirements.txt  # Python dependencies
├── .env              # Configuration for database and thresholds
└── README.md         # This file
```

-----

## Installation and Setup

1.  **Clone the repository** or download the files into a directory on your machine.

2.  **Create and activate a Python virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install Python dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate the synthetic dataset** (if not already present). This step samples the provided credit card dataset to create 5,000 transactions with random users, merchants and timestamps and then splits it into training, testing and simulation sets:

    ```bash
    python prepare_data.py
    ```

5.  **Seed the database** with the training data. This populates the `transactions` table with the training set, computes a risk score and classification for each row using a logistic regression model and writes the result to the database:

    ```bash
    python backend/seed_db.py
    ```

6.  **Configure the database** (optional). The application uses an SQLite database by default, which requires no additional setup. To use a MySQL database instead, edit the `.env` file and set `DB_URI` to a valid SQLAlchemy connection string. For example:

    ```env
    DB_URI=mysql+pymysql://user:password@localhost:3306/frauddb
    ```

    Ensure that the MySQL server is running and the target database exists before starting the backend.

7.  **Run the backend server**:

    ```bash
    python backend/app.py
    ```

    The server will start on `http://127.0.0.1:5000` by default. You should see log messages indicating that the Flask app has started.

8.  **Open the frontend**. With the backend running, open your web browser and navigate to `http://127.0.0.1:5000`. The page loads automatically from the Flask static file handler. You can select a user from the dropdown and click “Simulate Transaction” to see how the backend classifies the generated transactions.

-----

## Customisation

  * **Number of users**: Adjust the `populateUserSelect` call in `frontend/script.js` to change how many users are available in the dropdown. The data preparation script currently simulates 50 users.

  * **Classification thresholds**: The `.env` file defines the thresholds used to convert a probability into an Approved, Flagged or Fraud classification. Adjust `THRESH_APPROVE` and `THRESH_FLAG` as required and restart the server.

  * **Database engine**: While SQLite is used for convenience, the backend has been written using SQLAlchemy and will work with MySQL. Simply install a MySQL server, create a database, set `DB_URI` accordingly and reinstall the `pymysql` package if necessary. The `requirements.txt` already lists `pymysql` for this purpose.

  * **Retraining logic**: The backend retrains a logistic regression model on the user’s transaction history whenever a new transaction arrives, provided the user has at least 10 past transactions with both fraud and non‑fraud examples. Otherwise the global model (fit on the 3,500‑row training set) is used.

-----

## Notes and Limitations

  * The machine learning model in this demo is intentionally simplistic. Only the transaction amount is used as a feature, so risk scores correlate strongly with transaction size. In a real fraud detection system many more features (such as frequency, merchant type and user behaviour patterns) would be used.

  * The random data generation in `prepare_data.py` assigns merchants uniformly at random and picks user IDs from 1–50. Feel free to modify the merchant list or the range of user IDs to suit your own testing scenarios.

  * The provided `test_data.csv` split is not currently used by the backend but is included should you wish to measure model performance separately.

  * The web frontend is intentionally lightweight and does not require building with a toolchain. It uses vanilla JavaScript and communicates with the backend via the Fetch API.