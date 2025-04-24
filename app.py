from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
from datetime import datetime
from werkzeug.utils import secure_filename
from src.model_utils import load_model
from src.action_detector import detect_action
from config import Config

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MODEL_PATH'] = 'models/yolov8n.pt'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the model
model = load_model()
model_loaded = model is not None

if not model_loaded:
    print("Warning: Model failed to load. The system will run in demo mode.")

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
def intro():
    return render_template('intro.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)
            
        file = request.files['image']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            try:
                # Create a secure filename
                filename = secure_filename(file.filename)
                # Add timestamp to make filename unique
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # Ensure the upload directory exists
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                # Save the file
                file.save(file_path)
                
                # Detect action in the image
                if not model_loaded:
                    # Demo mode - return dummy results
                    flash('Model not loaded - running in demo mode')
                    return render_template('display_image.html', 
                                    filename=filename,
                                    prediction='Normal',
                                    confidence=100.0,
                                    probabilities={
                                        'Normal': 100.0,
                                        'Peaking': 0.0,
                                        'Sneaking': 0.0,
                                        'Stealing': 0.0
                                    })
                
                # Process the image using the model
                action_data = detect_action(model, file_path)
                if action_data:
                    image, detections, predicted_action, action_scores, bar_chart = action_data
                    
                    # Pass the action_scores directly as probabilities after converting to percentages
                    probabilities = {k: v * 100 for k, v in action_scores.items()}
                    
                    return render_template('display_image.html', 
                                        filename=filename,
                                        prediction=predicted_action[0],
                                        confidence=predicted_action[1] * 100,
                                        probabilities=probabilities)
                else:
                    flash('No objects detected in the image')
                    return redirect(request.url)
                    
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)
        else:
            flash('Invalid file type. Please upload an image file (jpg, jpeg, png)')
            return redirect(request.url)
            
    return render_template('image_upload.html')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/display_image/<filename>')
def display_image(filename):
    return render_template('display_image.html', filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=False)



