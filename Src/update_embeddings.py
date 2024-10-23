import os
import numpy as np
from features import load_embeddings_and_metadata, compute_embeddings, save_embeddings_and_metadata
from data_loader import load_metadata

# Đường dẫn đến file metadata và embeddings hiện có
current_dir = os.path.dirname(__file__)  # Thư mục hiện tại chứa file Python này
existing_metadata_file = os.path.join(current_dir, '../model/metadata.pkl')
existing_embeddings_file = os.path.join(current_dir, '../model/embeddings.pkl')

# Đường dẫn tương đối đến thư mục chứa ảnh mới
new_source_dir = os.path.join(current_dir, '../NewData')

def update_embeddings(existing_metadata_file, existing_embeddings_file, new_source_dir):
    # Tải metadata và embeddings cũ
    embeddings, metadata = load_embeddings_and_metadata(existing_embeddings_file, existing_metadata_file)

    # Tải metadata mới
    new_metadata = load_metadata(new_source_dir)

    # Lấy các đường dẫn đã có trong metadata cũ
    existing_paths = {meta.image_path() for meta in metadata}  # Giả sử `path` là thuộc tính của IdentityMetadata

    # Lọc ra metadata mới không trùng lặp
    filtered_new_metadata = [meta for meta in new_metadata if meta.image_path() not in existing_paths]

    # Nếu không có ảnh mới nào, thoát hàm
    if not filtered_new_metadata:
        print("Không có ảnh mới nào để cập nhật.")
        return

    # Tính toán embeddings cho ảnh mới
    new_embeddings = compute_embeddings(filtered_new_metadata)

    # Kết hợp dữ liệu cũ và mới
    updated_metadata = np.concatenate((metadata, filtered_new_metadata))
    updated_embeddings = np.concatenate((embeddings, new_embeddings), axis=0)

    # Lưu lại dữ liệu mới
    save_embeddings_and_metadata(updated_embeddings, updated_metadata, existing_embeddings_file, existing_metadata_file)
    print("Đã cập nhật embeddings và metadata thành công.")

if __name__ == "__main__":
    update_embeddings(existing_metadata_file, existing_embeddings_file, new_source_dir)
