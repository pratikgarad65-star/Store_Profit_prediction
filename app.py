import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "gradientboosting_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found in the current directory.")

# Feature names derived from your serialized model snippet
FEATURES = [
    "Ship Mode", "Customer Name", "Segment", "Country", 
    "City", "State", "Postal Code", "Region", 
    "Category", "Sub-Category", "Product Name", 
    "Sales", "Quantity", "Discount"
]

NUMERIC_FEATURES = ["Postal Code", "Sales", "Quantity", "Discount"]

# Embedded HTML/CSS/JS Template for an Interactive UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gradient Boosting Predictor</title>
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
            max-width: 900px;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-color);
            overflow: hidden;
        }

        .header {
            padding: 2rem;
            background: linear-gradient(135deg, #312e81, #4338ca);
            text-align: center;
        }

        .header h1 {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: #c7d2fe;
            font-size: 0.95rem;
        }

        form {
            padding: 2rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
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

        .form-group input {
            background-color: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-color);
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }

        .form-group input:focus {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3);
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1rem;
            background-color: var(--accent-color);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 1rem;
            font-size: 1rem;
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
            font-size: 1.1rem;
            color: var(--success-color);
            margin-bottom: 0.5rem;
        }

        .result-card .prediction-value {
            font-size: 2rem;
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
        <h1>Gradient Boosting Inference Portal</h1>
        <p>Enter feature parameters to obtain model output estimations</p>
    </div>

    <form id="predictionForm">
        <div class="grid">
            {% for feature in features %}
            <div class="form-group">
                <label for="{{ feature }}">{{ feature }}</label>
                <input 
                    type="{% if feature in numeric_features %}number{% else %}text{% endif %}" 
                    {% if feature in numeric_features %}step="any"{% endif %}
                    id="{{ feature }}" 
                    name="{{ feature }}" 
                    placeholder="Enter {{ feature }}" 
                    required
                >
            </div>
            {% endfor %}
            <button type="submit" class="btn-submit">Generate Prediction</button>
        </div>
    </form>

    <div id="resultCard" class="result-card">
        <h2>Predicted Output</h2>
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
                
                predictionValue.innerText = typeof result.prediction === 'number' 
                    ? result.prediction.toFixed(4) 
                    : result.prediction;
                    
                resultCard.style.display = 'block';
            } else {
                alert('Error: ' + result.error);
            }
        } catch (err) {
            alert('An error occurred while connecting to the server.');
        }
    });
</script>

</body>
</html>
"""

def process_features(raw_data):
    """
    Transforms raw input values into an array matching model structure.
    Encodes strings into numerical hash identifiers.
    """
    processed = []
    for feature in FEATURES:
        val = raw_data.get(feature, "")
        if feature in NUMERIC_FEATURES:
            try:
                processed.append(float(val))
            except ValueError:
                processed.append(0.0)
        else:
            # Hash string features into integers to prevent ML model failure
            processed.append(float(abs(hash(str(val))) % (10 ** 8)))
    return np.array([processed])

@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        HTML_TEMPLATE, 
        features=FEATURES, 
        numeric_features=NUMERIC_FEATURES
    )

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model pickle file is missing from server."}), 500

    try:
        raw_data = request.get_json()
        input_array = process_features(raw_data)
        
        # Execute model prediction
        prediction = model.predict(input_array)
        output = float(prediction[0])
        
        return jsonify({"prediction": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
