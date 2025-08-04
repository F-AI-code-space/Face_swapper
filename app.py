import os, sys
sys.path.append(os.path.abspath(__file__))

import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory, session 
import onnxruntime as ort
import time
import urllib

# Custom face swap module
import swap

# Define the folder for storing uploaded source face images
UPLOAD_FOLDER = r'Source_face_images'

# Dictionary to cache source images (key: image path, value: image data)
SOURCE_IMGS = {} 

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__, static_folder=r'static')

# Set a secret key for session management (IMPORTANT: use a strong key in production)
app.secret_key = 'your_super_secret_key'

# Route: Home page
@app.route('/')
def index():
    html = render_template('index.html')
    print(html)
    print("\n"*10)
    return html 

# Route: Store selected source face image
@app.route('/source_img', methods=['POST'])
def selected():
    data = request.get_json()

    # Extract filename from image URL and decode any URL-encoded characters
    img_url_relative = data.get('image_url').split('/')[-1]
    img_url_relative = urllib.parse.unquote(img_url_relative)

    # Construct full file path
    img_path = os.path.join(UPLOAD_FOLDER, img_url_relative)
    
    # Store image path in session
    session['source_img_path'] = img_path

    # Load and cache the source image using OpenCV
    print(img_path)
    SOURCE_IMGS[img_path] = cv2.imread(img_path)

    if img_path:
        print(f"URL is saved in: {img_path}")
        return jsonify({'message': 'Success', 'received_url': img_path}), 200
    else:
        return jsonify({'error': 'URL Error'}), 400

# Route: Remove image from upload folder
@app.route('/rmv_img', methods=['POST'])
def rmv():
    data = request.get_json()

    # Extract and decode image filename
    img_url_relative = data.get('image_url').split('/')[-1]
    img_url_relative = urllib.parse.unquote(img_url_relative)
    img_path = os.path.join(UPLOAD_FOLDER, img_url_relative)

    # Delete the file from filesystem
    sta = os.remove(img_path)
    
    if not sta:
        # If deleted image was the active session's source image, remove it from session
        if 'source_img_path' in session and session['source_img_path'] == img_path:
            session.pop('source_img_path', None)
        return jsonify({'message': "OK"}), 200
    else:
        return jsonify({"error": "faile"}), 400

# Route: Perform face swap on the provided image
@app.route('/swap_face', methods=['POST'])
def swap_face():
    start = time.time()

    # Decode base64 image from request
    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Retrieve cached source image from session
    source_img_path = session.get('source_img_path')
    source_img = SOURCE_IMGS.get(source_img_path, None)

    if source_img is not None:
        print(f"Executing swap with session-specific source image: {source_img_path}.....")
        img = swap.main(img, source_img)  # Face swap happens here
    else:
        print("No source image found in session or path invalid. Skipping swap.")

    # Encode image back to base64 for frontend display
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Return result image and FPS (swap speed)
    return jsonify({'image': "data:image/jpeg;base64," + img_base64, "fps": round(1 / (time.time() - start))})

# Route: Upload a new image to the server
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    # Accept only image files with allowed extensions
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        return 'Image uploaded successfully!', 200
    return 'Invalid file format', 400

# Route: Serve an image from the upload folder
@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Route: List all uploaded image filenames
@app.route('/list_images')
def list_images():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
    return jsonify(files)

# Entry point: Start Flask app
if __name__ == '__main__':
    try:
        # Attempt to run HTTPS server using SSL certificates
        # Note: cert.pem and key.pem must be present in the root directory
        app.run(host='0.0.0.0', port=4433, debug=True, threaded=True, ssl_context=('cert.pem', 'key.pem'))
    except FileNotFoundError:
        # Fallback to HTTP if SSL certs are not found
        print("Error: cert.pem or key.pem not found. Running on HTTP instead.")
        app.run(host='0.0.0.0', port=4434, debug=True, threaded=True)