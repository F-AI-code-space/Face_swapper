import os, sys
sys.path.append(os.path.abspath(__file__))

import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory, session 
import onnxruntime as ort
import time
import urllib

import swap

UPLOAD_FOLDER = r'Source_face_images'
SOURCE_IMGS = {} 
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder=r'static')
app.secret_key = 'your_super_secret_key' # !!! IMPORTANT: Set a strong, random secret key for production !!!

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/source_img', methods=['POST'])
def selected():
    data = request.get_json()
    img_url_relative = data.get('image_url').split('/')[-1]
    img_url_relative = urllib.parse.unquote(img_url_relative)
    img_path = os.path.join(UPLOAD_FOLDER, img_url_relative)
    
    # Store the path to the source image in the session
    session['source_img_path'] = img_path
    print(img_path)
    SOURCE_IMGS[img_path] = cv2.imread(img_path)

    if img_path:
        print(f"URL is saved in: {img_path}")
        return jsonify({'message': 'Success', 'received_url': img_path}), 200
    else:
        return jsonify({'error': 'URL Error'}), 400

@app.route('/rmv_img', methods=['POST'])
def rmv():
    data = request.get_json()
    img_url_relative = data.get('image_url').split('/')[-1]
    img_url_relative = urllib.parse.unquote(img_url_relative)
    img_path = os.path.join(UPLOAD_FOLDER, img_url_relative)
    sta = os.remove(img_path)
    
    if not sta:
        # If the removed image was the source image for this session, clear it
        if 'source_img_path' in session and session['source_img_path'] == img_path:
            session.pop('source_img_path', None)
        return jsonify({'message': "OK"}), 200
    else:
        return jsonify({"error": "faile"}), 400

@app.route('/swap_face', methods=['POST'])
def swap_face():
    start = time.time()
    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    source_img_path = session.get('source_img_path')
    source_img = SOURCE_IMGS.get(source_img_path, None)
    if source_img is not None:
        print(f"Executing swap with session-specific source image: {source_img_path}.....")
        img = swap.main(img, source_img) 
    else:
        print("No source image found in session or path invalid. Skipping swap.")

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': "data:image/jpeg;base64," + img_base64, "fps":round(1/(time.time() - start))})

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))
        return 'Image uploaded successfully!', 200
    return 'Invalid file format', 400

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/list_images')
def list_images():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
    return jsonify(files)

if __name__ == '__main__':
    try:
        # For production, ensure your secret key is properly managed (e.g., from environment variables)
        app.run(host='0.0.0.0', port=4433, debug=True, threaded=True, ssl_context=('cert.pem', 'key.pem'))
    except FileNotFoundError:
        print("Error: cert.pem or key.pem not found. Running on HTTP instead.")
        app.run(host='0.0.0.0', port=4433, debug=True, threaded=True)