// List of merchants to use when generating random transactions
const merchants = [
  'Amazon', 'Walmart', 'Target', 'Starbucks', 'Uber', 'Netflix',
  'Apple', 'Google Play', 'McDonalds', 'Costco'
];

// Summary counts for chart
let summaryCounts = {
  Approved: 0,
  Flagged: 0,
  Fraud: 0
};

let summaryChart;

// Initialise the page once the DOM has loaded
document.addEventListener('DOMContentLoaded', () => {
  populateUserSelect(50);
  initChart();
  // Load initial transactions for the first user
  const initialUser = parseInt(document.getElementById('userSelect').value);
  loadTransactions(initialUser);
  // Bind events
  document.getElementById('simulateBtn').addEventListener('click', simulateTransaction);
  document.getElementById('userSelect').addEventListener('change', (e) => {
    const userId = parseInt(e.target.value);
    // Reset chart and table when switching users
    summaryCounts = { Approved: 0, Flagged: 0, Fraud: 0 };
    updateChart();
    loadTransactions(userId);
  });
});

/**
 * Populate the user selection dropdown with user IDs from 1 to n.
 *
 * @param {number} n - Number of users.
 */
function populateUserSelect(n) {
  const select = document.getElementById('userSelect');
  select.innerHTML = '';
  for (let i = 1; i <= n; i++) {
    const option = document.createElement('option');
    option.value = i;
    option.textContent = `User ${i}`;
    select.appendChild(option);
  }
}

/**
 * Initialise the Chart.js bar chart to display a summary of
 * classifications for the current user.
 */
function initChart() {
  const ctx = document.getElementById('summaryChart').getContext('2d');
  summaryChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Approved', 'Flagged', 'Fraud'],
      datasets: [{
        label: '# of Transactions',
        data: [0, 0, 0],
        backgroundColor: [
          'rgba(75, 192, 192, 0.6)',
          'rgba(255, 206, 86, 0.6)',
          'rgba(255, 99, 132, 0.6)'
        ],
        borderColor: [
          'rgba(75, 192, 192, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(255, 99, 132, 1)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            precision: 0
          }
        }
      }
    }
  });
}

/**
 * Update the chart with the current summary counts.
 */
function updateChart() {
  summaryChart.data.datasets[0].data = [
    summaryCounts.Approved,
    summaryCounts.Flagged,
    summaryCounts.Fraud
  ];
  summaryChart.update();
}

/**
 * Load transactions for a specific user and display them in the table.
 *
 * @param {number} userId - The user ID to fetch transactions for.
 */
async function loadTransactions(userId) {
  try {
    const response = await fetch(`/transactions?user_id=${userId}`);
    if (!response.ok) {
      throw new Error('Failed to load transactions');
    }
    const transactions = await response.json();
    // Reset summary counts
    summaryCounts = { Approved: 0, Flagged: 0, Fraud: 0 };
    populateTable(transactions);
    // Update summary counts based on loaded data
    transactions.forEach(tx => {
      if (summaryCounts.hasOwnProperty(tx.result)) {
        summaryCounts[tx.result] += 1;
      }
    });
    updateChart();
  } catch (err) {
    console.error(err);
    alert('Error loading transactions');
  }
}

/**
 * Populate the HTML table with a list of transactions.
 *
 * @param {Array<object>} transactions - Array of transaction objects.
 */
function populateTable(transactions) {
  const tbody = document.querySelector('#transactionsTable tbody');
  // Clear existing rows
  tbody.innerHTML = '';
  transactions.forEach(tx => {
    addTableRow(tx);
  });
}

/**
 * Append a single transaction row to the table and update summary counts.
 *
 * @param {object} tx - A transaction record returned from the backend.
 */
function addTableRow(tx) {
  const tbody = document.querySelector('#transactionsTable tbody');
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${tx.id}</td>
    <td>${tx.user_id}</td>
    <td>${tx.merchant}</td>
    <td>${tx.amount.toFixed(2)}</td>
    <td>${new Date(tx.timestamp).toLocaleString()}</td>
    <td>${(tx.risk_score * 100).toFixed(2)}%</td>
    <td class="${tx.result.toLowerCase()}">${tx.result}</td>
  `;
  tbody.prepend(tr);
  // Update summary count and chart
  if (summaryCounts.hasOwnProperty(tx.result)) {
    summaryCounts[tx.result] += 1;
    updateChart();
  }
}

/**
 * Generate a random transaction payload and send it to the backend.
 */
async function simulateTransaction() {
  const userId = parseInt(document.getElementById('userSelect').value);
  // Generate random amount between $1 and $200
  const amount = parseFloat((Math.random() * 199 + 1).toFixed(2));
  const merchant = merchants[Math.floor(Math.random() * merchants.length)];
  const timestamp = new Date().toISOString();
  const payload = {
    user_id: userId,
    amount: amount,
    merchant: merchant,
    timestamp: timestamp
  };
  try {
    const response = await fetch('/simulate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || 'Unknown error');
    }
    const data = await response.json();
    // Append the new transaction to the table
    addTableRow(data.transaction);
  } catch (err) {
    console.error(err);
    alert('Error simulating transaction: ' + err.message);
  }
}