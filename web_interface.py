from flask import Flask, request, render_template, send_file, redirect, url_for
import os
from werkzeug.utils import secure_filename
import subprocess
import sys
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image using our adversarial_ml.py script
        try:
            result_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}.png")
            
            # Call the test_custom_image function in the main script
            cmd = ["python", "adversarial_ml.py", "test_image", filepath]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                # If there was an error in the subprocess
                return render_template('index.html', 
                                      error=f'Error processing image: {result.stderr}')
            
            # Parse output to get results
            output = result.stdout
            
            # Extract safety status
            if "is SAFE" in output:
                is_safe = True
                safety_message = "✅ The image is SAFE under adversarial attack."
            else:
                is_safe = False
                safety_message = "❌ The image is NOT SAFE; adversarial attack changed the prediction."
            
            # Return the results page
            return render_template('result.html',
                                  original_filename=filename,
                                  result_image=f"result_{filename}.png",
                                  is_safe=is_safe,
                                  safety_message=safety_message)
            
        except Exception as e:
            return render_template('index.html', 
                                  error=f'Server error processing image: {str(e)}')
    else:
        return render_template('index.html', error='File type not allowed')

@app.route('/results/<filename>')
def results(filename):
    return send_file(os.path.join(app.config['RESULTS_FOLDER'], filename))

@app.route('/uploads/<filename>')
def uploads(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/evaluate')
def evaluate():
    try:
        # Run the evaluation script
        cmd = ["python", "adversarial_ml.py", "evaluate"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return render_template('index.html', 
                                  error=f'Error evaluating model: {result.stderr}')
        
        # Return the evaluation results
        return render_template('evaluate.html', 
                              metrics_image='performance_metrics.png',
                              output=result.stdout)
        
    except Exception as e:
        return render_template('index.html', 
                              error=f'Server error during evaluation: {str(e)}')

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create the HTML templates if they don't exist
def create_templates():
    # Index page
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Adversarial Image Tester</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .error { color: red; font-weight: bold; }
        .button { background-color: #3498db; color: white; padding: 10px 15px; border: none; 
                 border-radius: 4px; cursor: pointer; }
        .button:hover { background-color: #2980b9; }
        .nav { margin: 20px 0; }
        .nav a { margin-right: 15px; color: #3498db; text-decoration: none; }
        .nav a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Adversarial Image Resilience Tester</h1>
    
    <div class="nav">
        <a href="/">Home</a>
        <a href="/evaluate">Evaluate Model</a>
    </div>
    
    <div class="container">
        <p>Upload an image to test if it's resilient against adversarial attacks.</p>
        <p>The model will analyze your image and determine if its classification remains consistent when adversarial noise is applied.</p>
        
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <input type="submit" value="Test Image" class="button">
        </form>
        
        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}
    </div>
    
    <div class="container">
        <h2>About Adversarial Attacks</h2>
        <p>Adversarial attacks are subtle modifications to images that are imperceptible to humans but can fool AI models into making incorrect predictions.</p>
        <p>This tool tests your images against the Fast Gradient Sign Method (FGSM) attack and evaluates whether the model maintains its original prediction.</p>
    </div>
</body>
</html>''')
    
    # Results page
    if not os.path.exists('templates/result.html'):
        with open('templates/result.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Adversarial Test Results</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .safe { color: green; font-weight: bold; }
        .unsafe { color: red; font-weight: bold; }
        img { max-width: 100%; }
        .button { background-color: #3498db; color: white; padding: 10px 15px; border: none; 
                 border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        .button:hover { background-color: #2980b9; }
        .nav { margin: 20px 0; }
        .nav a { margin-right: 15px; color: #3498db; text-decoration: none; }
        .nav a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Adversarial Test Results</h1>
    
    <div class="nav">
        <a href="/">Home</a>
        <a href="/evaluate">Evaluate Model</a>
    </div>
    
    <div class="container">
        <h2>Image: {{ original_filename }}</h2>
        
        {% if is_safe %}
        <p class="safe">{{ safety_message }}</p>
        {% else %}
        <p class="unsafe">{{ safety_message }}</p>
        {% endif %}
        
        <h3>Comparison of Original vs Adversarial Image</h3>
        <img src="{{ url_for('results', filename=result_image) }}" alt="Comparison">
        
        <p>The left image is your original upload, and the right image shows the same image with adversarial noise applied.</p>
        <p>The labels show the model's classification before and after the attack.</p>
    </div>
    
    <a href="/" class="button">Test Another Image</a>
</body>
</html>''')
    
    # Evaluation page
    if not os.path.exists('templates/evaluate.html'):
        with open('templates/evaluate.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Results</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #2c3e50; }
        .container { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        img { max-width: 100%; }
        .button { background-color: #3498db; color: white; padding: 10px 15px; border: none; 
                 border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        .button:hover { background-color: #2980b9; }
        .nav { margin: 20px 0; }
        .nav a { margin-right: 15px; color: #3498db; text-decoration: none; }
        .nav a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Model Evaluation Results</h1>
    
    <div class="nav">
        <a href="/">Home</a>
        <a href="/evaluate">Evaluate Model</a>
    </div>
    
    <div class="container">
        <h2>Performance Metrics</h2>
        <img src="{{ url_for('results', filename=metrics_image) }}" alt="Performance Metrics">
        
        <h2>Evaluation Output</h2>
        <pre>{{ output }}</pre>
    </div>
    
    <a href="/" class="button">Test an Image</a>
</body>
</html>''')

# Create templates on startup
create_templates()

if __name__ == '__main__':
    # Default port is 5000, but you can change it
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)