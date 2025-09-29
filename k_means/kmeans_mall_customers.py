# KMeans_MallCustomers.py
# =========================
# Thuật toán K-Means Clustering trên bộ dữ liệu Mall Customers
# =========================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1) Đọc dữ liệu
df = pd.read_csv("Mall_Customers.csv")

print("📊 5 dòng đầu của dữ liệu:")
print(df.head())

# 2) Chọn đặc trưng để phân cụm (Annual Income và Spending Score)
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) Elbow Method để chọn số cụm k
sse = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    sse.append(km.inertia_)   # inertia = tổng SSE

plt.figure(figsize=(6,4))
plt.plot(K, sse, "bx-")
plt.xlabel("Số cụm k")
plt.ylabel("SSE (Sum of Squared Errors)")
plt.title("Elbow Method để chọn số cụm k")
plt.show()

# 4) Huấn luyện KMeans với k=5 (chọn từ Elbow)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 5) Đánh giá Silhouette Score
score = silhouette_score(X_scaled, df["Cluster"])
print(f"\n🔎 Silhouette Score: {score:.3f}")

# 6) Vẽ cụm khách hàng
plt.figure(figsize=(8,6))
for cluster in range(5):
    cluster_points = df[df["Cluster"] == cluster]
    plt.scatter(cluster_points["Annual Income (k$)"],
                cluster_points["Spending Score (1-100)"],
                label=f"Cluster {cluster}")

# Vẽ tâm cụm
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1],
            s=200, c="black", marker="X", label="Centroids")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Phân cụm khách hàng bằng K-Means (Income & Score)")
plt.legend()
plt.show()

# 7) In kết quả (không hiển thị CustomerID nữa)
print("\n📌 Một vài khách hàng sau khi phân cụm:")
print(df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head(10))
