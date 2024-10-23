import os
import cv2
import numpy as np

class IdentityMetadata():
    def __init__(self, base, name, file):
        self.base = base
        self.name = name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            ext = os.path.splitext(f)[1]
            if ext.lower() in ['.jpg', '.jpeg']:
                metadata.append(IdentityMetadata(path, i, f))  # Giả sử bạn đã định nghĩa IdentityMetadata
    return np.array(metadata)

    return np.array(metadata)
def load_image(path):
    img = cv2.imread(path, 1)
    if img is None:
        raise ValueError(f"Không thể tải ảnh từ đường dẫn: {path}")  # Thêm thông báo lỗi
    return img[..., ::-1]  # Chuyển đổi từ BGR sang RGB

