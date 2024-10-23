import os
import cv2
from mtcnn import MTCNN
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from model import vgg_face
from data_loader import load_metadata, load_image
from features import compute_embeddings, compute_embedding_for_image, load_embeddings_and_metadata, save_embeddings_and_metadata
from classifier import train_classifier, predict
from utils import display_prediction

# Đường dẫn dữ liệu
current_dir = os.path.dirname(__file__)  # Thư mục hiện tại chứa file Python này
metadata_file = os.path.join(current_dir, '../model/metadata.pkl')
embeddings_file = os.path.join(current_dir, '../model/embeddings.pkl')

source_dir = os.path.join(current_dir, '../105_classes_pins_dataset')

# Kiểm tra xem có sẵn embeddings và metadata không
if os.path.exists(embeddings_file) and os.path.exists(metadata_file):
    embeddings, metadata = load_embeddings_and_metadata(embeddings_file, metadata_file)
else:
    metadata = load_metadata(source_dir)
    embeddings = compute_embeddings(metadata)
    save_embeddings_and_metadata(embeddings, metadata)


# Chia dữ liệu
train_idx = np.arange(metadata.shape[0]) % 9 != 0
test_idx = np.arange(metadata.shape[0]) % 9 == 0
X_train = embeddings[train_idx]
X_test = embeddings[test_idx]
targets = np.array([m.name for m in metadata])
y_train = targets[train_idx]
y_test = targets[test_idx]

# Chuẩn hóa
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Áp dụng PCA
pca = PCA(n_components=128)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
# Huấn luyện mô hình
clf, le = train_classifier(X_train_pca, y_train)

# Dự đoán
accuracy, predictions = predict(clf, le, X_test_pca, y_test)
print(predictions)
print(y_test)
print(f'Accuracy: {accuracy:.2f}')

new_image_path = os.path.join(current_dir, '../NewData/Tuan/DucTuan_10_face_1.jpg')


new_image = load_image(new_image_path)  # Giả định hàm này trả về ảnh đã được chuẩn hóa
img = load_image(new_image_path)
# Chuyển đổi ảnh mới thành mảng NumPy nếu cần
if isinstance(new_image, list):
    new_image = np.array(new_image)

# Đảm bảo ảnh có hình dạng (1, height, width, channels)
if new_image.ndim == 3:  # Nếu ảnh chỉ có (height, width, channels)
    new_image = np.expand_dims(new_image, axis=0)  # Thêm chiều batch

# Tính toán embedding cho tấm ảnh mới
new_embedding = compute_embedding_for_image(new_image[0])  # Chỉ lấy ảnh đầu tiên

# Chuẩn hóa tấm ảnh mới
new_embedding_std = scaler.transform([new_embedding])  # Chuyển đổi thành danh sách để chuẩn hóa

# Áp dụng PCA cho tấm ảnh mới
new_embedding_pca = pca.transform(new_embedding_std)

# Dự đoán cho tấm ảnh mới
new_prediction = clf.predict(new_embedding_pca)

# Giải mã nhãn dự đoán
predicted_label = le.inverse_transform(new_prediction)

# Hiển thị kết quả
print(f'Predicted label for the new image: {predicted_label[0]}')

display_prediction(img, predicted_label[0])
