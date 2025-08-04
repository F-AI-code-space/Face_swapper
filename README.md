# Face Swapper

## Purpose and Application

**Face Swapper** is a web-based application designed to demonstrate the power—and risks—of modern AI face manipulation technologies. This project is part of the "AI thật AI giả" talkshow, an initiative aimed at raising public awareness about deepfakes, their creative potential, and the ethical concerns they pose.

Whether you're a developer, educator, journalist, or curious citizen, this tool allows you to explore face swapping technology in a transparent and responsible way.

## Introduction

Face Swapper is a Python-based web application that enables users to swap faces in images quickly and efficiently. The backend is powered by machine learning models (ONNX), while the frontend offers a clean and intuitive web interface. It supports both CPU and GPU environments, making it accessible and performant across different systems.

## Features

- Face Swapping: Upload two faces and swap them instantly.
- Web UI: Clean, minimal interface with drag-and-drop support.
- Fast Processing: Uses ONNX models for speed and compatibility.
- Cross-Platform: Runs on CPU or GPU.
- HTTPS Ready: Optional SSL support for secure deployment.

## Requirements

- Python 3.7 or higher
- Flask
- OpenCV
- PyTorch
- ONNX Runtime
- HTML/CSS (Frontend)

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/F-AI-code-space/Face_swapper.git
cd Face_swapper
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install PyTorch

PyTorch is not included in `requirements.txt` due to system-specific installation steps.

- Visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- Choose your OS, Python version, and compute platform (CPU or CUDA).

Example for CPU:

```bash
pip install torch torchvision torchaudio
```

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Python Requirements

```bash
pip install -r requirements.txt
```

### 5. Install ONNX Runtime (CPU or GPU)

For GPU:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

For CPU:

```bash
pip install onnxruntime
```

More info: [https://onnxruntime.ai/docs/install/](https://onnxruntime.ai/docs/install/)

### 6. Optional: Enable HTTPS (Self-signed SSL)

To securely serve the app (e.g., in public demos or talks):

Generate certificates:

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

Then modify `app.py`:

```python
app.run(host='0.0.0.0', port=4433, ssl_context=('cert.pem', 'key.pem'))
```

Note: For production, consider using Let's Encrypt instead of self-signed certificates.

## Run the Application
In app.py, you need to set this key for security purpose
```python
app.secret_key = 'your_super_secret_key'
```


To run code
```bash
python app.py
```

Then open:

- [http://localhost:4434](http://localhost:4434) (HTTP)
- [https://localhost:4433](https://localhost:4433) (HTTPS, if SSL is configured)

## Acknowledgements

We would like to express our sincere gratitude to the authors of the [DeepMirror](https://github.com/MonsieurNam/DeepMirror) project for generously sharing their source code and allowing us to use it as a reference. Their work provided valuable insights and support for the development of this project.

## License

MIT License

## Contribution
- [Nguyễn Thành Công](https://github.com/VNthcong520712)
- [Châu Quốc Inh](https://github.com/inhcqce190593)

Happy face swapping — and stay aware of what AI can (and should) do!