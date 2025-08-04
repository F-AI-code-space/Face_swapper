import torch
import sys
import os
import insightface
from insightface.app import FaceAnalysis  

# Define the default model storage path for InsightFace
insightface_model_root = os.path.join(os.path.expanduser('~'), '.insightface', 'models')

# Create model directory if it doesn't exist
if not os.path.exists(insightface_model_root):
	os.makedirs(insightface_model_root)
	print(f"InsightFace/models is created: {insightface_model_root}")

# Check if CUDA (GPU support) is available
cuda = torch.cuda.is_available()
if cuda:
	print('Using:', torch.cuda.get_device_name())  # Print GPU name
	providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']  # Use GPU first, fallback to CPU
else:
	print('Using: CPU')
	providers = ['CPUExecutionProvider']  # Use CPU only

# Initialize the face analysis model (face detector, landmark extractor, etc.)
app = FaceAnalysis(name='buffalo_l', providers=providers)

# Prepare the model:
# - ctx_id=0 means use GPU
# - ctx_id=-1 means use CPU
# - det_size is the target input size for detection (width, height)
app.prepare(ctx_id=cuda - 1, det_size=(640, 640))

# Load the face swapper model (ONNX format)
swapper = insightface.model_zoo.get_model(
	'inswapper_128.onnx',
	providers=providers
)

# Main function to perform face swap
# - frame: input image where faces will be replaced
# - source: image containing the face to be used for swapping
def main(frame, source=None):
	# Detect face(s) in the source image
	source_faces = app.get(source)
	if len(source_faces) == 0:
		print("Error: Source face not found")
		sys.exit(1)  # Exit if no face found in source image

	# Use the first detected face in the source as the swap target
	source_face = source_faces[0]

	# Detect face(s) in the target (frame) image
	target_faces = app.get(frame)

	# If target faces are found, perform the swap
	if len(target_faces) > 0:
		for target_face in target_faces:
			# Perform face swap and paste the result back onto the original frame
			frame = swapper.get(frame, target_face, source_face, paste_back=True)
		return frame
	else:
		# No face found in target image
		print("Cannot swap!!!!!!")
		return frame
