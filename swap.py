import torch
import sys
import os
import insightface
from insightface.app import FaceAnalysis

insightface_model_root = os.path.join(os.path.expanduser('~'), '.insightface', 'models')
if not os.path.exists(insightface_model_root):
	os.makedirs(insightface_model_root)
	print(f"InsightFace/models is created: {insightface_model_root}")

cuda = torch.cuda.is_available()
if cuda:
	print('Using:', torch.cuda.get_device_name())
	providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
else:
	print('Using: CPU')
	providers=['CPUExecutionProvider']
app = FaceAnalysis(name='buffalo_l', providers = providers)

# ctx_id=0 is GPU, ctx_id=-1 is CPU.
app.prepare(ctx_id=cuda-1, det_size=(640, 640)) 

swapper = insightface.model_zoo.get_model(
	'inswapper_128.onnx',
	providers = providers
	)

def main(frame, source=None):
	source_faces = app.get(source)
	if len(source_faces) == 0:
		print("Error: Source face not found")
		sys.exit(1)
	source_face = source_faces[0]

	target_faces = app.get(frame)

	if len(target_faces) > 0:
		for target_face in target_faces:
			frame = swapper.get(frame, target_face, source_face, paste_back=True)

		return frame
	else:
		print("Cannot swap!!!!!!")
		return frame
