import cv2
import numpy as np
import sys
import os
import insightface
from insightface.app import FaceAnalysis

# --- Cấu hình và Tải mô hình ---
# Đảm bảo bạn đã cài đặt thư viện insightface và onnxruntime:
# pip install insightface onnxruntime opencv-python numpy

# QUAN TRỌNG: Nếu bạn gặp lỗi tải xuống tự động mô hình, vui lòng tải xuống thủ công:
# 1. Mô hình phân tích khuôn mặt ('antelope'):
#    Tải xuống: https://github.com/deepinsight/insightface/releases/download/v0.7/antelope.zip
#    Sau khi tải về, giải nén tệp 'antelope.zip'. Bạn sẽ nhận được một thư mục 'antelope'.
#    ĐẶT THƯ MỤC 'antelope' NÀY VÀO: C:\Users\thanh\.insightface\models\

# 2. Mô hình hoán đổi khuôn mặt ('inswapper_128.onnx'):
#    Tải xuống: https://github.com/deepinsight/insightface/releases/download/v0.3/inswapper_128.onnx
#    ĐẶT TỆP 'inswapper_128.onnx' NÀY VÀO: C:\Users\thanh\.insightface\models\inswapper\
#    (Bạn có thể cần tạo thư mục 'inswapper' nếu nó chưa tồn tại)

# Đảm bảo đường dẫn .insightface\models tồn tại
insightface_model_root = os.path.join(os.path.expanduser('~'), '.insightface', 'models')
if not os.path.exists(insightface_model_root):
	os.makedirs(insightface_model_root)
	print(f"Đã tạo thư mục mô hình InsightFace: {insightface_model_root}")

print("Đang khởi tạo mô hình InsightFace...")
# Khởi tạo FaceAnalysis để phát hiện khuôn mặt và trích xuất đặc trưng
# 'antelope' là tên bộ mô hình mặc định cho các tác vụ phân tích khuôn mặt.
# Providers: ['CPUExecutionProvider'] là mặc định. Nếu có GPU, bạn có thể thử ['CUDAExecutionProvider', 'CPUExecutionProvider']
providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
app = FaceAnalysis(name='buffalo_l', providers = providers)

# Chuẩn bị ứng dụng FaceAnalysis. ctx_id=0 cho GPU, ctx_id=-1 cho CPU.
# det_size là kích thước ảnh mà bộ phát hiện khuôn mặt sẽ làm việc, lớn hơn cho khuôn mặt nhỏ hơn.
app.prepare(ctx_id=0, det_size=(640, 640)) 

# Tải mô hình hoán đổi khuôn mặt
# Đảm bảo 'inswapper_128.onnx' được đặt đúng cách như hướng dẫn ở trên
swapper = insightface.model_zoo.get_model(
	'inswapper_128.onnx',
	providers = providers
	)
print("Khởi tạo mô hình hoàn tất.")

# --- Logic chính của ứng dụng ---

def main(frame, source=None):
	# Tải ảnh nguồn (khuôn mặt bạn muốn hoán đổi vào)
	# QUAN TRỌNG: Thay thế 'your_source_face.jpg' bằng đường dẫn thực tế đến tệp ảnh của bạn.
	# Để có kết quả tốt nhất, hãy sử dụng ảnh có khuôn mặt rõ ràng, chính diện.
	# source_img_path = '/mnt/hdd/Lab_data/Lab_code/Faceswap/Faceswaper/app/images/thuy_diem.webp'
	
	# if not os.path.exists(source_img_path):
	# 	print(f"Lỗi: Không tìm thấy ảnh nguồn tại {source_img_path}")
	# 	print("Vui lòng thay thế 'your_source_face.jpg' bằng đường dẫn đến tệp ảnh chứa khuôn mặt.")
	# 	print("Đảm bảo ảnh nằm trong cùng thư mục với script này hoặc cung cấp đường dẫn đầy đủ.")
	# 	sys.exit(1)

	# source_img = cv2.imread(source_img_path)
	# if source_img is None:
	# 	print(f"Lỗi: Không thể tải ảnh nguồn từ {source_img_path}. Kiểm tra định dạng tệp hoặc bị hỏng.")
	# 	sys.exit(1)
	
	# Phát hiện khuôn mặt trong ảnh nguồn. Đây là "khuôn mặt ID" để hoán đổi.
	source_faces = app.get(source)
	if len(source_faces) == 0:
		print("Lỗi: Không tìm thấy khuôn mặt trong ảnh nguồn. Vui lòng cung cấp ảnh rõ ràng.")
		sys.exit(1)
	# Lấy khuôn mặt đầu tiên làm khuôn mặt nguồn
	source_face = source_faces[0]

	# Khởi tạo webcam để quay video thời gian thực.
	# '0' thường là webcam mặc định. Thay đổi nếu bạn có nhiều camera.
	# cap = cv2.VideoCapture(0)

	# if not cap.isOpened():
	#     print("Lỗi: Không thể mở luồng video. Kiểm tra xem webcam có được kết nối hoặc đang được ứng dụng khác sử dụng không.")
	#     sys.exit(1)

	# print("\n--- Kích hoạt hoán đổi khuôn mặt thời gian thực với InsightFace ---")
	# print("Đang quay từ camera PC...")
	# print("Nhấn 'q' để thoát ứng dụng.")

	# while True:
	# Đọc một khung hình từ camera. Đây là trọng tâm của xử lý thời gian thực.
	# ret, frame = cap.read()
	# if not ret:
	# 	print("Không thể lấy khung hình. Đang thoát...")
	# 	break

	# Lật khung hình theo chiều ngang để có góc nhìn gương tự nhiên hơn, phổ biến cho webcam.
	# frame = cv2.flip(frame, 1)

	# Phát hiện tất cả các khuôn mặt trong khung hình hiện tại (các khuôn mặt mục tiêu).
	target_faces = app.get(frame)

	# Hoán đổi khuôn mặt cho mỗi khuôn mặt được phát hiện trong khung hình.
	# insightface cho phép hoán đổi nhiều khuôn mặt cùng lúc.
	if len(target_faces) > 0:
		for target_face in target_faces:
			# Thực hiện hoán đổi khuôn mặt.
			# 'paste_back=True' đảm bảo khuôn mặt đã hoán đổi được dán trở lại vào khung hình.
			frame = swapper.get(frame, target_face, source_face, paste_back=True)
		
		# (Tùy chọn) Vẽ hình chữ nhật xung quanh khuôn mặt đã phát hiện nếu bạn muốn trực quan hóa
		# for face in target_faces:
		#     bbox = face.bbox.astype(np.int32)
		#     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
		
		# cv2.imshow('Hoán đổi khuôn mặt thời gian thực (Nhấn Q để thoát)', frame)
		return frame
	else:
		# Nếu không tìm thấy khuôn mặt nào trong khung hình hiện tại, chỉ hiển thị khung hình gốc.
		# cv2.imshow('Hoán đổi khuôn mặt thời gian thực (Nhấn Q để thoát)', frame)
		print("Cannot swap!!!!!!")
		return frame

	# Kiểm tra phím 'q' để thoát vòng lặp và đóng ứng dụng.
	# cv2.waitKey(1) đợi 1 mili giây, cho phép cập nhật khung hình thời gian thực.
	# if cv2.waitKey(1) & 0xFF == ord('q'):
	# 	break

	# # Giải phóng tài nguyên webcam và đóng tất cả các cửa sổ OpenCV.
	# cap.release()
	# cv2.destroyAllWindows()

# if __name__ == "__main__":
# 	main()
