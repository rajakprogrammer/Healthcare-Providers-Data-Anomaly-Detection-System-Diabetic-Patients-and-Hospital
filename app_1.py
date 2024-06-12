# Import necessary libraries
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FloatField
from wtforms.validators import InputRequired, Length
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import os

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')
# In-memory user storage
users = {}

# Flask-WTF Form classes
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    submit = SubmitField('Login')

class AnalyzeInstanceForm(FlaskForm):
    pregnancies = FloatField('Pregnancies', validators=[InputRequired()])
    glucose = FloatField('Glucose', validators=[InputRequired()])
    blood_pressure = FloatField('Blood Pressure', validators=[InputRequired()])
    skin_thickness = FloatField('Skin Thickness', validators=[InputRequired()])
    insulin = FloatField('Insulin', validators=[InputRequired()])
    bmi = FloatField('BMI', validators=[InputRequired()])
    diabetes_pedigree_function = FloatField('Diabetes Pedigree Function', validators=[InputRequired()])
    age = FloatField('Age', validators=[InputRequired()])
    submit = SubmitField('Analyze')

# Load and preprocess the dataset
def load_dataset(file):
    df = pd.read_csv(file)
    names = df['Name']
    features = df.drop(columns=['Name'])
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, scaler, names, df.columns

# Build the autoencoder model
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Train the autoencoder model
def train_autoencoder(X_train):
    autoencoder = build_autoencoder(X_train.shape[1])
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)
    return autoencoder

# Detect anomalies using the autoencoder model
def detect_anomalies(model, X, scaler, threshold):
    X_pred = model.predict(X)
    mse = ((X - X_pred) ** 2).mean(axis=1)
    anomalous_indices = [i for i, error in enumerate(mse) if error > threshold]
    return mse, scaler.inverse_transform(X_pred), anomalous_indices

# Generate threshold value
def generate_threshold(anomalies):
    threshold = anomalies.max() * 1  # Adjust multiplier as needed
    return threshold

# Define the home route
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session['username']
    return render_template('index.html', username=username)

# Define the registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if username in users:
            return 'User already exists!'
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        users[username] = hashed_password
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

# Define the login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('index'))
        return 'Invalid username or password!'
    return render_template('login.html', form=form)

# Define the logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Define the analyze dataset route
@app.route('/analyze_dataset', methods=['POST'])
def analyze_dataset():
    if 'username' not in session:
        return redirect(url_for('login'))

    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided.'}), 400

    # Load and preprocess the entire dataset
    df, scaler, names, columns = load_dataset(file)

    # Train the autoencoder model on the entire dataset
    autoencoder = train_autoencoder(df)

    # Initially predict the reconstructed data to generate anomalies for the threshold
    # This is where you had the error; we'll skip this step and move directly to threshold generation

    # Detect anomalies using the autoencoder model
    anomalies, _, _ = detect_anomalies(autoencoder, df, scaler, 0)  # Temporary call to get anomalies

    # Generate threshold value based on initial anomaly detection
    threshold = generate_threshold(anomalies)

    a = 0.05

    # Now, properly detect anomalies using the autoencoder model with the correct threshold value
    anomalies, reconstructed_data, anomalous_indices = detect_anomalies(autoencoder, df, scaler, a)

    # Get the names of anomalous patients
    anomalous_names = [names.iloc[i] for i in anomalous_indices]

    # Prepare data for rendering in the template
    results = []
    for i, name in enumerate(names):
        result = {
            'name': name,
            'anomaly': i in anomalous_indices
        }
        results.append(result)

    return render_template('results.html', results=results, threshold=a)

# Define the analyze instance route
@app.route('/analyze_instance', methods=['GET', 'POST'])
def analyze_instance():
    form = AnalyzeInstanceForm()
    if form.validate_on_submit():
        # Collect input data from the form
        instance_data = [
            form.pregnancies.data,
            form.glucose.data,
            form.blood_pressure.data,
            form.skin_thickness.data,
            form.insulin.data,
            form.bmi.data,
            form.diabetes_pedigree_function.data,
            form.age.data
        ]
        # Extracting form data and ensuring they are floats
        a = float(form.pregnancies.data)
        b = float(form.glucose.data)
        c = float(form.blood_pressure.data)
        d = float(form.skin_thickness.data)
        e = float(form.insulin.data)
        f = float(form.bmi.data)
        g = float(form.diabetes_pedigree_function.data)
        h = float(form.age.data)

        # Creating a dictionary to form a DataFrame
        data = {
            'column1': [a],
            'column2': [b],
            'column3': [c],
            'column4': [d],
            'column5': [e],
            'column6': [f],
            'column7': [g],
            'column8': [h]
            }

        # Creating the DataFrame
        instance_df = pd.DataFrame(data)
        scaler = MinMaxScaler()
        instance_scaled = scaler.fit_transform(instance_df)

        # Load the pre-trained autoencoder model (assuming you have a saved model)
        autoencoder = train_autoencoder(instance_scaled)

        # Use a pre-determined threshold for anomaly detection (adjust as needed)
        threshold = 0.20

        # Detect anomalies for the single instance
        mse, _, _ = detect_anomalies(autoencoder, instance_scaled, scaler, threshold)

        # Determine if the instance is anomalous
        is_anomalous = mse[0] > threshold

        # Redirect to results page
        return render_template('11.html', is_anomalous=is_anomalous, mse=mse[0])
    return render_template('10.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)