🛒 Mall Customers Clustering - Flask App

Ứng dụng web nhỏ sử dụng Flask + Scikit-learn + Bootstrap 5 để thực hiện phân cụm khách hàng dựa trên dữ liệu Mall Customers.

📂 Cấu trúc thư mục

<img width="318" height="222" alt="image" src="https://github.com/user-attachments/assets/67bb43c7-774e-49f5-90aa-875a48496b3f" />


⚙️ Yêu cầu
Python 3.8+
Các thư viện:
pip install flask pandas scikit-learn matplotlib

🚀 Cách chạy
Clone hoặc tải project về.
Đặt file Mall_Customers.csv vào thư mục gốc cùng app.py.
Chạy ứng dụng:
python app.py

Mở trình duyệt tại địa chỉ:
http://127.0.0.1:5000/

📊 Chức năng
Xem 5 dòng đầu tiên của dữ liệu khách hàng.
Thực hiện phân cụm KMeans (mặc định k = 5).
Hiển thị biểu đồ scatter plot (Annual Income vs Spending Score).
Xem Silhouette Score để đánh giá chất lượng phân cụm.
Hiển thị bảng kết quả (Cluster của từng khách hàng).
🎨 Giao diện
Sử dụng Bootstrap 5 để hiển thị đẹp, responsive.
Có navbar, card, và bảng dữ liệu rõ ràng.
📷 Demo

Trang chủ:
<img width="1914" height="670" alt="image" src="https://github.com/user-attachments/assets/2bcc7ffe-4c8e-4c7c-a1f3-a56b4569fd56" />
Kết quả phân cụm:
<img width="1904" height="976" alt="image" src="https://github.com/user-attachments/assets/5d84a8c3-4c6b-484c-beda-a02969d5c72a" />

