### Tóm tắt

Nhận dạng khuôn mặt là một lĩnh vực quan trọng trong trí tuệ nhân tạo, và sự kết hợp giữa **VGGFace** và **MTCNN** mang lại giải pháp mạnh mẽ cho bài toán này. 

**VGGFace** là mô hình mạng nơ-ron sâu được đào tạo trên tập dữ liệu lớn, có khả năng tạo ra các biểu diễn đặc trưng cho từng khuôn mặt. Trong khi đó, **MTCNN** là một mạng nơ-ron chuyên phát hiện khuôn mặt và xác định các điểm đặc trưng như mắt, mũi, và miệng.

Khi kết hợp VGGFace với MTCNN, MTCNN sẽ phát hiện và định vị khuôn mặt trong ảnh, sau đó VGGFace sẽ trích xuất các đặc trưng để nhận diện. Sự kết hợp này không chỉ nâng cao độ chính xác mà còn cải thiện hiệu suất xử lý trong các điều kiện thực tế khác nhau.

### Cách sử dụng code

1. **Cài đặt thư viện**  
   Đầu tiên, bạn cần cài đặt các thư viện cần thiết cho việc sử dụng VGGFace và MTCNN. Bạn có thể cài đặt chúng bằng pip:

   ```bash
   pip install -r requirements.txt
   ```
   **Lưu ý:** để sử dụng code, cần tải mô hình vgg_face_weight.h5 tại [đây]([URL](https://drive.google.com/file/d/1xhZue6xMcQ-ZWyvv-deiVMMex8cphuUx/view?usp=sharing))
 
