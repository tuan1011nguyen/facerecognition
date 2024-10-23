### Tóm tắt

Nhận dạng khuôn mặt là một lĩnh vực quan trọng trong trí tuệ nhân tạo, và sự kết hợp giữa **VGGFace** và **MTCNN** mang lại giải pháp mạnh mẽ cho bài toán này. 

**VGGFace** là mô hình mạng nơ-ron sâu được đào tạo trên tập dữ liệu lớn, có khả năng tạo ra các biểu diễn đặc trưng cho từng khuôn mặt. Trong khi đó, **MTCNN** là một mạng nơ-ron chuyên phát hiện khuôn mặt và xác định các điểm đặc trưng như mắt, mũi, và miệng.

Khi kết hợp VGGFace với MTCNN, MTCNN sẽ phát hiện và định vị khuôn mặt trong ảnh, sau đó VGGFace sẽ trích xuất các đặc trưng để nhận diện. Sự kết hợp này không chỉ nâng cao độ chính xác mà còn cải thiện hiệu suất xử lý trong các điều kiện thực tế khác nhau.

### Cách hoạt động của MTCNN

**MTCNN** (Multi-task Cascaded Convolutional Networks) là mô hình phát hiện khuôn mặt thông qua 3 bước:

1. **P-Net**: Quét ảnh với nhiều tỉ lệ khác nhau, phát hiện các vùng có khả năng chứa khuôn mặt.
2. **R-Net**: Lọc và tinh chỉnh các bounding boxes do P-Net tạo ra để loại bỏ các vùng không phải khuôn mặt.
3. **O-Net**: Tinh chỉnh cuối cùng và xác định các điểm mốc (landmarks) trên khuôn mặt như mắt, mũi, miệng.

### Cách hoạt động của VGGFace

**VGGFace** là mạng nơ-ron sâu dùng để trích xuất các đặc trưng (embeddings) từ khuôn mặt. Sau khi MTCNN phát hiện khuôn mặt, VGGFace nhận diện khuôn mặt bằng cách:

1. **Tiền xử lý ảnh**: Chuyển đổi ảnh về kích thước chuẩn (224x224) và chuẩn hóa dữ liệu.
2. **Trích xuất đặc trưng**: Sử dụng mô hình VGGFace (thường là ResNet50) để tạo ra các vector đặc trưng đại diện cho khuôn mặt.
3. **So sánh khuôn mặt**: Các vector đặc trưng này được so sánh để xác định liệu hai khuôn mặt có giống nhau hay không dựa trên khoảng cách cosine hoặc Euclidean.


### Cách sử dụng code

1. **Cài đặt thư viện**  
   Đầu tiên, bạn cần cài đặt các thư viện cần thiết cho việc sử dụng VGGFace và MTCNN. Bạn có thể cài đặt chúng bằng pip:

   ```bash
   pip install -r requirements.txt
   ```
   **Lưu ý:** để sử dụng code, cần tải mô hình vgg_face_weight.h5 tại ([đây](https://drive.google.com/file/d/1xhZue6xMcQ-ZWyvv-deiVMMex8cphuUx/view?usp=sharing)). Sau đó hãy chèn mô hình vào thư mục model/.

2. **Cách thêm dữ liệu mới**
   Để thêm dữ liệu mới mà không cần training lại mô hình, chúng tôi sử dụng embeddings để lưu trữ các giá trị đặc trưng của ảnh bằng Vggface chỉ có các lớp Conv, pooling mà không có lớp Fully Connected.
   Tại folder NewData, tạo một folder có tên của người cần thêm vào và trong đó chứa các ảnh của người đó
   Mở file Src/update_embeddings.py và **Run**

3. **Thực hiện việc dự đoán**
   Tại ```bash main.py``` hãy thay đổi đường dẫn của *new_image_path* và **Run** để thực hiện dự đoán
   
 
