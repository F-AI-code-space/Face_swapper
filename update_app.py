# import os, sys
# import cv2
# import numpy as np
# import base64
# from flask import Flask, render_template, request, jsonify, send_from_directory
# import torch 
# import onnxruntime as ort
# import time

# import testswap

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# UPLOAD_FOLDER = 'images'
# SOURCE_IMG = ''
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# sys.path.append(os.path.abspath(__file__))

# # Check GPU availability at startup
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU Name:", torch.cuda.get_device_name(0))
# print("ONNX Runtime providers:", ort.get_available_providers(), '\n\n')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# app = Flask(__name__, static_folder='static')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/source_img', methods=['POST'])
# def selected():
#     global SOURCE_IMG
#     data = request.get_json()
#     img_url = "images/"+data.get('image_url').split('/')[-1]
#     SOURCE_IMG = cv2.imread(img_url)
#     print(SOURCE_IMG.shape)

#     if img_url:
#         print(f"URL ảnh được chọn từ client: {img_url}")
#         return jsonify({'message': 'URL đã được nhận thành công!', 'received_url': img_url}), 200
#     else:
#         return jsonify({'error': 'Không nhận được URL ảnh nào.'}), 400

# @app.route('/rmv_img', methods=['POST'])
# def rmv():
#     data = request.get_json()
#     img_url = "images/"+data.get('image_url').split('/')[-1]
#     sta = os.system(f'rm {img_url}')
#     print(sta)
#     if not sta:
#         return jsonify({'message': "OK"}), 200
#     else: 
#         return jsonify({"error": "faile"}), 400


# @app.route('/detect_face', methods=['POST'])
# def detect_face():
#     start = time.time()
#     global SOURCE_IMG
#     data = request.json
#     img_data = base64.b64decode(data['image'].split(',')[1])
#     nparr = np.frombuffer(img_data, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     if not isinstance(SOURCE_IMG, str):
#         print("Excuting swap.....")
#         img = testswap.main(img, SOURCE_IMG)
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#     # for (x, y, w, h) in faces:
#     #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     # # Encode lại ảnh đã vẽ box
#     _, buffer = cv2.imencode('.jpg', img)
#     img_base64 = base64.b64encode(buffer).decode('utf-8')
#     return jsonify({'image': "data:image/jpeg;base64," + img_base64, "fps":round(1/(time.time() - start))})

# @app.route('/upload', methods=['POST'])
# def upload():
# 	file = request.files['image']
# 	if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
# 		file.save(os.path.join(UPLOAD_FOLDER, file.filename))
# 		return 'Image uploaded successfully!', 200
# 	return 'Invalid file format', 400

# @app.route('/images/<filename>')
# def get_image(filename):
# 	return send_from_directory(UPLOAD_FOLDER, filename)

# @app.route('/list_images')
# def list_images():
# 	files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
# 	return jsonify(files)

# if __name__ == '__main__':
#     try:
#         app.run(host='0.0.0.0', port=4433, debug=True, threaded=True, ssl_context=('cert.pem', 'key.pem'))
#     except FileNotFoundError:
#         print("Error: cert.pem or key.pem not found. Running on HTTP instead.")
#         app.run(host='0.0.0.0', port=4433, debug=True, threaded=True)


import os, sys
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory, session # Import session
import torch
import onnxruntime as ort
import time
import urllib

import update_swap as testswap

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
UPLOAD_FOLDER = 'images'
SOURCE_IMGS = {} # Remove this global variable
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
sys.path.append(os.path.abspath(__file__))

# Check GPU availability at startup
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
print("ONNX Runtime providers:", ort.get_available_providers(), '\n\n')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_super_secret_key' # !!! IMPORTANT: Set a strong, random secret key for production !!!

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/source_img', methods=['POST'])
def selected():
    # global SOURCE_IMG # No longer needed
    data = request.get_json()
    img_url_relative = data.get('image_url').split('/')[-1]
    img_url_relative = urllib.parse.unquote(img_url_relative)
    img_path = os.path.join(UPLOAD_FOLDER, img_url_relative)
    
    # Store the path to the source image in the session
    session['source_img_path'] = img_path
    print(img_path)
    SOURCE_IMGS[img_path] = cv2.imread(img_path)
    
    # It's better to load the image when needed, or store the image data itself
    # For now, let's just store the path and load it in detect_face
    
    if img_path:
        print(f"URL ảnh được chọn từ client và lưu vào session: {img_path}")
        return jsonify({'message': 'URL đã được nhận thành công!', 'received_url': img_path}), 200
    else:
        return jsonify({'error': 'Không nhận được URL ảnh nào.'}), 400

@app.route('/rmv_img', methods=['POST'])
def rmv():
    data = request.get_json()
    img_url_relative = data.get('image_url').split('/')[-1]
    img_url_relative = urllib.parse.unquote(img_url_relative)
    img_path = os.path.join(UPLOAD_FOLDER, img_url_relative)
    sta = os.system(f'rm "{img_path}"') # Consider using os.remove for better error handling
    print(sta)
    if not sta:
        # If the removed image was the source image for this session, clear it
        if 'source_img_path' in session and session['source_img_path'] == img_path:
            session.pop('source_img_path', None)
        return jsonify({'message': "OK"}), 200
    else:
        return jsonify({"error": "faile"}), 400


# ... (các import và khai báo khác không thay đổi)

@app.route('/detect_face', methods=['POST'])
def detect_face():
    start = time.time()
    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    source_img_path = session.get('source_img_path') # Lấy đường dẫn ảnh nguồn từ session
    source_img = SOURCE_IMGS.get(source_img_path, None)
    if source_img is not None:
        print(f"Executing swap with session-specific source image: {source_img_path}.....")
        # Truyền đường dẫn ảnh nguồn vào testswap.main
        img = testswap.main(img, source_img) 
    else:
        print("No source image found in session or path invalid. Skipping swap.")
        # Bạn có thể trả về một thông báo lỗi hoặc hình ảnh gốc ở đây
        # return jsonify({'error': 'No source image selected for this session.'}), 400

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({'image': "data:image/jpeg;base64," + img_base64, "fps":round(1/(time.time() - start))})

# ... (các route khác không thay đổi)


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