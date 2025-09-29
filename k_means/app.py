from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # dùng backend không cần GUI
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io
import base64

app = Flask(__name__)

# Đọc dữ liệu 1 lần
df = pd.read_csv("Mall_Customers.csv")

@app.route("/")
def home():
    # Hiển thị 5 dòng đầu
    preview = df.head().to_html(classes="table table-bordered table-striped", index=False)
    return render_template("index.html", preview=preview)

@app.route("/clusters")
def clusters():
    # Chọn đặc trưng
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans với k=5
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered["Cluster"] = kmeans.fit_predict(X_scaled)

    # Silhouette Score
    score = silhouette_score(X_scaled, df_clustered["Cluster"])

    # Vẽ scatter plot
    plt.figure(figsize=(8,6))
    for cluster in range(5):
        cluster_points = df_clustered[df_clustered["Cluster"] == cluster]
        plt.scatter(cluster_points["Annual Income (k$)"],
                    cluster_points["Spending Score (1-100)"],
                    label=f"Cluster {cluster}")

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1],
                s=200, c="black", marker="X", label="Centroids")

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("KMeans Clustering (Income & Spending Score)")
    plt.legend()

    # Lưu ảnh vào bộ nhớ
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    # Bảng kết quả
    table = df_clustered[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head(15).to_html(classes="table table-bordered table-striped", index=False)

    return render_template("clusters.html", score=score, plot_url=img_base64, table=table)

if __name__ == "__main__":
    app.run(debug=True)

