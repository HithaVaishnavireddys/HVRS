from flask import Flask, request, render_template
import numpy as np
import joblib

# Initialize Flask app
caseapp = Flask(__name__)

# Load pre-trained stacking ensemble model and label encoder
model = joblib.load('stacking_rf_xgb_lgb_tuned.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Manual encoding maps (must match training)
protocol_map = {'udp': 0, 'tcp': 1, 'icmp': 2}
service_map = {
    'private': 0, 'ftp_data': 1, 'eco_i': 2, 'telnet': 3, 'http': 4,
    'smtp': 5, 'ftp': 6, 'ldap': 7, 'pop_3': 8, 'courier': 9
}
flag_map = {
    'REJ': 0, 'SF': 1, 'RSTO': 2, 'S0': 3, 'RSTR': 4,
    'SH': 5, 'S3': 6, 'S2': 7, 'S1': 8, 'RSTOS0': 9
}

# Feature names (must match training order)
feature_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

@caseapp.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@caseapp.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Collect and encode input
        features = []
        for feat in feature_names:
            val = request.form.get(feat, '')

            if feat == 'protocol_type':
                val = protocol_map.get(val, -1)
            elif feat == 'service':
                val = service_map.get(val, -1)
            elif feat == 'flag':
                val = flag_map.get(val, -1)
            else:
                val = float(val) if val else 0.0

            features.append(val)

        features_array = np.array([features])

        # Step 2: Predict probabilities
        probs = model.predict_proba(features_array)[0]
        top3 = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:3]
        top_preds = [f"{label_encoder.inverse_transform([cls])[0]} ({prob:.2%})" for cls, prob in top3]
        result = f'Top Predictions: {", ".join(top_preds)}'

        # Step 3: Interpret system status
        normal_conf = next((prob for cls, prob in top3 if label_encoder.inverse_transform([cls])[0] == 'normal'), 0) * 100
        if normal_conf > 95:
            status = "✅ Your system is safe. No signs of attack detected. For peace of mind, you can still contact the helpline: 1930."
        elif normal_conf > 90:
            status = "⚠️ System appears normal but shows suspicious behavior. Monitor closely or contact 1930."
        elif normal_conf > 80:
            status = "⚠️ Elevated risk detected. Stay alert and consider contacting 1930."
        elif normal_conf > 75:
            status = "⚠️ High risk of breach. Contact 1930 for assistance."
        elif normal_conf > 50:
            status = "⚠️ Equal chance of breach. Better to contact 1930 immediately."
        else:
            status = "🚨 Your system is likely under attack. Take immediate action and call 1930."

        return render_template('index.html', prediction_text=result, status_message=status)

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    caseapp.run(debug=True)
