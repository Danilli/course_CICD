import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import DBSCAN
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('../data/scaled_data.csv', index_col = "kkt_id")

embeded_folder = '../data/embeddings/'

embede_dict = {}

# Проходим по всем файлам в папке
for filename in os.listdir(embeded_folder):
    if filename.endswith('embeddings.csv'):  
        emb_name = os.path.splitext(filename)[0]  # Получаем имя модели без расширения
        emb_path = os.path.join(embeded_folder, filename)  # Полный путь к модели
        
        # Загружаем модель с помощью TensorFlow/Keras
        embedingers = pd.read_csv(emb_path, index_col = "kkt_id")
        
        # Добавляем модель в словарь
        embede_dict[emb_name] = embedingers

def apply_pca(dfa, centroids, n_components=2):
    
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(dfa)
    centroids = pca.transform(centroids)
    # Создание нового DataFrame с результатами PCA
    columns = [f'PC{i+1}' for i in range(n_components)]
    df_result = pd.DataFrame(data=df_pca, columns=columns)
    #print(df_result)
    return df_result, centroids

def pic_kmeans(kmeans, dat):
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    #dt = apply_pca(dat)
    dt, centroids = apply_pca(dat, centroids)
    
    plt.scatter(dt["PC1"], dt["PC2"], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300, c='red', label='Centroids')
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

def silhoette_graph(kmeans, cluster_labels, X, n_clusters):
    
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    # Создание графика
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 7)
    
    # График Силуэта
    y_lower = 10
    for i in range(n_clusters):
        # Собираем силуэты для точек в кластере i
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
    
        # Наносим на график метку кластера
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        # Вычисляем новую y_lower для следующего графика
        y_lower = y_upper + 10  # 10 для разделения между кластерами
    
    ax1.set_title(f"График метрики Силуэта для {kmeans.n_clusters} кластеров")
    ax1.set_xlabel("Значение метрики Силуэта")
    ax1.set_ylabel("Метка кластера")
    
    # Линия для среднего значения Силуэта
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    ax1.set_yticks([])  # Убираем yticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    plt.show()


def extract_centroids(model, model_number, df):
    # Получаем центроиды из модели
    centroids = model.cluster_centers_
    
    # Создаем датафрейм для хранения информации о центроидах
    centroid_df = pd.DataFrame(centroids, columns=df.columns)
    
    # Добавляем информацию о принадлежности к кластеру и номере модели
    centroid_df['cluster'] = range(model.n_clusters)
    centroid_df['model_number'] = model_number
    
    return centroid_df


def vis_centroid_graph(centroids):
    unique_models = centroids['model_number'].unique()
    
    for model in unique_models:
        subset = centroids[centroids['model_number'] == model]
        subset = subset.drop('model_number', axis = 1)
        
        # Создание графика
        plt.figure(figsize=(10, 6))
        
        for idx, row in subset.iterrows():
            plt.plot(row[2:-1], label=f'Cluster {row["cluster"]}')
        
        plt.title(f'Model Number {model}')
        #plt.xlabel('Date')
        #plt.ylabel('Feature Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def do_10_claster_models(df):
    init_method = 'k-means++'  # Использование k-means++ для инициализации центроидов
    max_iter = 300  # Максимальное количество итераций
    tol=1e-5
    means_df = pd.DataFrame(index=df.index)
    centroids = pd.DataFrame()
    centroids_data = []
    for n_clusters in range(2,11):
        # Создание модели k-средних
        kmeans = KMeans(n_clusters=n_clusters, init=init_method, tol=tol, max_iter=max_iter, random_state=45)
        cluster_labels = kmeans.fit_predict(df)
        
        means_df[f'cluster_{n_clusters}'] = cluster_labels
        
        f = extract_centroids(kmeans, n_clusters, df)
        centroids = pd.concat([f, centroids], axis=0)
        
        pic_kmeans(kmeans, df)
        silhoette_graph(kmeans, cluster_labels, df, n_clusters)

    vis_centroid_graph(centroids)

# Проходим по каждому датафрейму в словаре
for name, df in embede_dict.items():
    print(name)
    do_10_claster_models(df)




def dbscan_pca_vis(datafr, eps=0.9, min_samples=50, metric='euclidean'):
    """
    Функция для кластеризации данных с использованием DBSCAN и визуализации с помощью PCA.

    Параметры:
    - datafr: pd.DataFrame, данные для кластеризации и визуализации.
    - eps: float, максимальное расстояние между двумя точками для их объединения в кластер.
    - min_samples: int, минимальное количество точек для формирования кластера.
    - metric: str, метрика расстояния.

    Возвращает:
    - pd.DataFrame с метками кластеров.
    """
    # Кластеризация с использованием DBSCAN
    df_dbscan_test = dbscan_predict(datafr, eps=eps, min_samples=min_samples, metric=metric)

    # Применение PCA для визуализации
    pca = PCA(n_components=2)
    array_pca_vis = pca.fit_transform(datafr)
    df_pca_vis = pd.DataFrame(data=array_pca_vis, index=datafr.index, columns=['Component_1', 'Component_2'])

    # Визуализация кластеров
    plot_labels(df_pca_vis, df_dbscan_test)

    return df_dbscan_test


def dbscan_predict(df: pd.DataFrame, eps=0.9, min_samples=50, metric='euclidean'):
    """
    Функция для кластеризации данных с использованием DBSCAN.

    Параметры:
    - df: pd.DataFrame, данные для кластеризации.
    - eps: float, максимальное расстояние между двумя точками для их объединения в кластер.
    - min_samples: int, минимальное количество точек для формирования кластера.
    - metric: str, метрика расстояния.

    Возвращает:
    - pd.DataFrame с метками кластеров.
    """
    X = df.values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=16)
    dbscan.fit(X)
    labels = dbscan.labels_

    return pd.DataFrame(data=labels, index=df.index, columns=['db_scan_label'])


def plot_labels(df_two_dim: pd.DataFrame, df_res: pd.DataFrame, n_objects: int = 5000):
    """
    Функция для визуализации кластеров на двумерной плоскости.

    Параметры:
    - df_two_dim: pd.DataFrame, двумерные данные для визуализации.
    - df_res: pd.DataFrame, результаты кластеризации с метками.
    - n_objects: int, количество объектов для визуализации.
    """
    label_cols = [col for col in df_res.columns if 'label' in col]
    data_two_dimensional_df = df_two_dim.copy()

    if len(label_cols) > 1:
        raise ValueError("Функция поддерживает только одну колонку с метками.")

    col_ans = label_cols[0]
    fig, ax = plt.subplots(figsize=(11, 8))
    data_two_dimensional_df['class'] = df_res[col_ans]

    sns.scatterplot(
        data=data_two_dimensional_df[:n_objects],
        x='Component_1', y='Component_2',
        hue='class', palette='Set1', alpha=0.6, ax=ax
    )
    ax.set_title(f'Распределение точек ({col_ans})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True)
    plt.show()

def cluster_and_visualize(datafr, eps=0.9, min_samples=50, metric='euclidean', n_objects=5000):
    """
    Функция для последовательного выполнения кластеризации DBSCAN, PCA и визуализации.

    Параметры:
    - datafr: pd.DataFrame, данные для кластеризации и визуализации.
    - eps: float, максимальное расстояние между двумя точками для их объединения в кластер.
    - min_samples: int, минимальное количество точек для формирования кластера.
    - metric: str, метрика расстояния.
    - n_objects: int, количество объектов для визуализации.

    Возвращает:
    - pd.DataFrame с метками кластеров.
    """
    # Кластеризация данных с использованием DBSCAN
    df_dbscan_test = dbscan_predict(datafr, eps=eps, min_samples=min_samples, metric=metric)

    # Применение PCA для визуализации
    pca = PCA(n_components=2)
    array_pca_vis = pca.fit_transform(datafr)
    df_pca_vis = pd.DataFrame(data=array_pca_vis, index=datafr.index, columns=['Component_1', 'Component_2'])

    # Визуализация кластеров
    plot_labels(df_pca_vis, df_dbscan_test, n_objects=n_objects)

    return df_dbscan_test

def get_kdist_plot(df: pd.DataFrame,
                   k = None,
                   radius_nbrs: float = 1.0):
    """Параметр radius_nbrs не влияет на работу алгоритма KNN"""

    if k is None:
        k = df.shape[1] * 2 - 1

    print(f"Number of neighbors = {k}")

    # p = 2 для Евклидова расстояния 
    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs, p=2).fit(df)

    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(df) 
                                       
    distances = np.sort(distances, axis=0)
    distances = distances[:, k-1]

    # Plot the sorted K-nearest neighbor distance for each point in the dataset
    plt.figure(figsize=(8,8))
    plt.plot(distances)
    plt.xlabel('Points/Objects in the dataset', fontsize=12)
    plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    plt.close()

embede_dict.keys()

reluAll = embede_dict["reluAll_-92-64-32-16-12-16-32-64-92-_e150_b16_encoder_embeddings"]
sigmoidAll = embede_dict["sigmoidAll_-92-64-32-16-12-16-32-64-92-_e100_b40_encoder_embeddings"]
softsign_tanhEND = embede_dict["softsign_tanhEND_-64-32-16-32-64-_e300_b48_encoder_embeddings"]
LeakyReLU_tanhEND = embede_dict["LeakyReLU_tanhEND_-128-64-32-16-32-64-128-_e200_b32_encoder_embeddings"]

k = 2 * reluAll.shape[-1] - 1 # k=2*{dim(dataset)} - 1
get_kdist_plot(df=reluAll, k=k, radius_nbrs=1)

# Пример вызова функции
clustered_data = cluster_and_visualize(datafr=reluAll, 
                                       eps=1, 
                                       min_samples=30, 
                                       metric='euclidean', 
                                       n_objects=5000)

k = 2 * reluAll.shape[-1] - 1 # k=2*{dim(dataset)} - 1
get_kdist_plot(df=sigmoidAll, k=k, radius_nbrs=1)

# Пример вызова функции
clustered_data = cluster_and_visualize(datafr=sigmoidAll, 
                                       eps=0.0013, 
                                       min_samples=100, 
                                       metric='euclidean', 
                                       n_objects=5000)

k = 2 * reluAll.shape[-1] - 1 # k=2*{dim(dataset)} - 1
get_kdist_plot(df=softsign_tanhEND, k=k, radius_nbrs=1)


# Пример вызова функции
clustered_data = cluster_and_visualize(datafr=softsign_tanhEND, 
                                       eps=0.4, 
                                       min_samples=90, 
                                       metric='euclidean', 
                                       n_objects=5000)

k = 2 * reluAll.shape[-1] - 1 # k=2*{dim(dataset)} - 1
get_kdist_plot(df=LeakyReLU_tanhEND, k=k, radius_nbrs=1)

# Пример вызова функции
clustered_data = cluster_and_visualize(datafr=LeakyReLU_tanhEND, 
                                       eps=0.7, 
                                       min_samples=40, 
                                       metric='euclidean', 
                                       n_objects=5000)

k = 2 * df.shape[-1] - 1 # k=2*{dim(dataset)} - 1
get_kdist_plot(df=df, k=k, radius_nbrs=1)

# Пример вызова функции
clustered_data = cluster_and_visualize(datafr=df, 
                                       eps=9, 
                                       min_samples=200, 
                                       metric='euclidean', 
                                       n_objects=5000)


