import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

# ------------------------------------------------------------------
# TOP-LEVEL 'app' INSTANCE (Required by Vercel's Python runtime)
# ------------------------------------------------------------------
app = Flask(__name__)

# Load the trained model safely
MODEL_PATH = "gradientboosting_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")

# Features expected by your GradientBoostingRegressor model
FEATURES = [
    "Ship Mode", "Customer Name", "Segment", "Country", 
    "City", "State", "Postal Code", "Region", 
    "Category", "Sub-Category", "Product Name", 
    "Sales", "Quantity", "Discount"
]

# Multiple Choice (MCQ) Dropdown Options Mapping
OPTIONS = {
    "Ship Mode": ["Standard Class", "Second Class", "First Class", "Same Day"],
    "Customer Name": ["Retail Customer", "Corporate Client", "Wholesale Buyer", "Small Business"],
    "Segment": ["Consumer", "Corporate", "Home Office"],
    "Country": ["United States"],
    "City": ["New York City", "Los Angeles", "Chicago", "Houston", "Philadelphia", "Phoenix", "San Antonio"],
    "State": ["New York", "California", "Illinois", "Texas", "Pennsylvania", "Arizona"],
    "Postal Code": ["10001", "90001", "60601", "77001", "19101", "85001"],
    "Region": ["East", "West", "Central", "South"],
    "Category": ["Technology", "Furniture", "Office Supplies"],
    "Sub-Category": ["Phones", "Chairs", "Storage", "Tables", "Binders", "Accessories", "Paper", "Furnishings"],
    "Product Name": ["Standard Office Equipment", "Premium Technology Item", "Bulk Supplies Pack", "Executive Furniture Set"],
    "Sales": ["$100 - $500", "$500 - $1,000", "$1,000 - $2,500", "$2,500 - $5,000", "$5,000+"],
    "Quantity": ["1", "2", "3", "5", "10", "15"],
    "Discount": ["0% (No Discount)", "10% Discount", "20% Discount", "30% Discount", "50% Discount"]
}

# Numeric conversion mappings for Sales and Discount options
NUMERIC_MAPPINGS = {
    "Sales": {
        "$100 - $500": 300.0,
        "$500 - $1,000": 750.0,
        "$1,000 - $2,500": 1750.0,
        "$2,500 - $5,000": 3750.0,
        "$5,000+": 6000.0
    },
    "Discount": {
        "0% (No Discount)": 0.0,
        "10% Discount": 0.10,
        "20% Discount": 0.20,
        "30% Discount": 0.30,
        "50% Discount": 0.50
    }
}

# HTML, Interactive UI Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Store Profit Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --success-color: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 2rem;
        }

        .container {
            width: 100%;
            max-width: 950px;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-color);
            overflow: hidden;
        }

        .header {
            padding: 2.5rem;
            background: linear-gradient(135deg, #312e81, #4338ca);
            text-align: center;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: #c7d2fe;
            font-size: 1rem;
        }

        form {
            padding: 2rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.25rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .form-group select {
            background-color: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-color);
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .form-group select:focus {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1.5rem;
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 1rem;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s ease, transform 0.1s ease;
        }

        .btn-submit:hover {
            background-color: var(--accent-hover);
        }

        .btn-submit:active {
            transform: scale(0.99);
        }

        .result-card {
            margin: 0 2rem 2rem 2rem;
            padding: 1.5rem;
            border-radius: 12px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--success-color);
            text-align: center;
            display: none;
            animation: fadeIn 0.3s ease-in-out forwards;
        }

        .result-card h2 {
            font-size: 1.2rem;
            color: var(--success-color);
            margin-bottom: 0.5rem;
        }

        .result-card .prediction-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #ffffff;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Store Profit Prediction</h1>
        <p>Select parameters from the dropdown options to compute store profit</p>
    </div>

    <form id="predictionForm">
        <div class="grid">
            {% for feature in features %}
            <div class="form-group">
                <label for="{{ feature }}">{{ feature }}</label>
                <select id="{{ feature }}" name="{{ feature }}" required>
                    {% for opt in options[feature] %}
                    <option value="{{ opt }}">{{ opt }}</option>
                    {% endfor %}
                </select>
            </div>
            {% endfor %}
            <button type="submit" class="btn-submit">Predict Profit</button>
        </div>
    </form>

    <div id="resultCard" class="result-card">
        <h2>Estimated Store Profit</h2>
        <div id="predictionValue" class="prediction-value">--</div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const data = {};
        formData.forEach((value, key) => data[key] = value);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                const resultCard = document.getElementById('resultCard');
                const predictionValue = document.getElementById('predictionValue');
                
                const val = result.prediction;
                predictionValue.innerText = "$" + val.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    
                resultCard.style.display = 'block';
            } else {
                alert('Error: ' + result.error);
            }
        } catch (err) {
            alert('An error occurred connecting to the server.');
        }
    });
</script>

</body>
</html>
"""

def process_input(raw_data):
    """Encodes selected MCQ dropdown text into valid input feature floats."""
    processed = []
    for feature in FEATURES:
        val = raw_data.get(feature, "")
        
        if feature in NUMERIC_MAPPINGS:
            # Map selected ranges back to numerical values
            processed.append(NUMERIC_MAPPINGS[feature].get(val, 0.0))
        elif feature in ["Postal Code", "Quantity"]:
            try:
                processed.append(float(val))
            except ValueError:
                processed.append(0.0)
        else:
            # Consistent hash encoding for text values
            processed.append(float(abs(hash(str(val))) % (10 ** 8)))
            
    return np.array([processed])

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        HTML_TEMPLATE, 
        features=FEATURES, 
        options=OPTIONS
    )

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "gradientboosting_model.pkl file not loaded on server."}), 500

    try:
        raw_data = request.get_json()
        input_array = process_input(raw_data)
        
        prediction = model.predict(input_array)
        output = float(prediction[0])
        
        return jsonify({"prediction": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
