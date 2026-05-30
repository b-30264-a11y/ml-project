from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # --- Field names match exactly what HTML sends ---
        present_price = float(data['present_price'])
        kms_driven    = int(data['kms_driven'])
        car_age       = int(data['car_age'])        # HTML computes & sends this
        fuel_type     = int(data['fuel_type'])       # already encoded (0/1/2) from HTML
        seller_type   = int(data['seller_type'])     # already encoded (0/1) from HTML
        transmission  = int(data['transmission'])    # already encoded (0/1) from HTML
        owner         = int(data['owner'])


        # --- Build feature array in training order ---
        # Order: Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner, Car_Age
        features = np.array([[
            present_price,
            kms_driven,
            fuel_type,
            seller_type,
            transmission,
            owner,
            car_age
        ]])

        # --- Scale and predict ---
        features_scaled = scaler.transform(features)
        prediction = round(float(model.predict(features_scaled)[0]), 2)

        # --- Response keys match what HTML checks ---
        return jsonify({
            'success': True,             # HTML: if(data.success)
            'predicted_price': prediction  # HTML: data.predicted_price
        })

    except KeyError as e:
        return jsonify({'success': False, 'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)