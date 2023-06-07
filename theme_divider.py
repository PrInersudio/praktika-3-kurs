from sklearn.feature_extraction.text import TfidfVectorizer
import os
import librosa
import magic
import sys
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pymorphy3 import MorphAnalyzer
import string
from langdetect import detect
import iso639
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import kstest
from scipy.stats import uniform, norm, expon
from PIL import Image



def calculate_dist(sample):

    dist_list = [uniform, norm, expon]

    hist, bins = np.histogram(sample, bins=10)

    dist_prop = [np.mean(sample), np.std(sample)]
    dist_prop.extend(hist)
    dist_prop.extend(bins)

    choosed_dist = 0
    max_p_value = 0
    for i, dist in enumerate(dist_list):
        _, p_value = kstest(sample, dist.cdf)
        if p_value > max_p_value:
            max_p_value = p_value
            choosed_dist = i
    dist_prop.append(choosed_dist)

    return dist_prop

# Текст
def preprocess_text(text):
    # Определение языка текста
    lang_code = detect(text)
    lang_name = iso639.to_name(lang_code).lower()
    
    # Токенизация текста
    tokens = word_tokenize(text.lower(), lang_name)
    
    # Удаление знаков пунктуации
    tokens = [token for token in tokens if token not in string.punctuation]

    
    # Лемматизация слов
    if lang_name == 'russian':
        lemmatizer = MorphAnalyzer()
        tokens = [lemmatizer.parse(token)[0].normal_form for token in tokens]
    elif lang_name == 'english':
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    else:
        print("Текущая версия программы не поддерживает лемматизацию для языка " + lang_name + ". Дальнейшая обработка данного текста будет проведена без лемматизации.")
    
    # Выбор набора стоп-слов на основе языка
    stop_words = set(stopwords.words(lang_name))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Объединение токенов обратно в текстовую строку
    processed_text = ' '.join(tokens)
    
    return processed_text



# Изображение
def get_color_histogram(file_path):
    # Открываем картиику
    image = Image.open(file_path)
    # Убеждаемся, что все картинки будут в одном режиме цветности
    rgba = image.convert("RGBA")
    # Получаем гистограмму
    histogram = rgba.histogram()
    return histogram


def calculate_hog_features(image):

    # Парметры HOG
    win_size = (64, 128)  # Размер окна
    block_size = (16, 16)  # Размер блока
    block_stride = (8, 8)  # Шаг блока
    cell_size = (8, 8)  # Размер ячейки
    nbins = 9  # Количество направлений градиента

    # Приведение изображения к размеру окна
    resized_image = cv2.resize(image, win_size)
    
    # Вычисление гистограммы распределения направленных градиентов (HOG)
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hist = hog.compute(resized_image)
    
    # Нормализация гистограммы
    hist = hist.flatten()
    hist = hist / np.linalg.norm(hist)

    return hist


def calculate_texture_features(image):
    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Вычисление матрицы смежности градиентов (GLCM)
    distances = [1]  # расстояние между пикселями
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # углы для вычисления различных градиентов
    glcm = graycomatrix(gray_image, distances, angles, levels=256, symmetric=True, normed=True)
    
    # Вычисление текстурных признаков на основе GLCM
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    return np.concatenate((contrast, dissimilarity, homogeneity, energy, correlation))


def get_image_contours(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение алгоритма Кэнни
    edges = cv2.Canny(gray, 100, 200)
    # Контуры на изображении
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_flattened = []
    for contour in contours:
        contours_flattened.extend(contour.flatten())

    return calculate_dist(contours_flattened)



# Аудио
def get_audio_features(file_path):
    # Загрузка аудиофайла
    audio, sr = librosa.load(file_path)
    # Извлечение MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    # Извлечение хроматических признаков
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    # Извлечение темпа и ритмических признаков
    _, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    # Извлечение спектрального контраста
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    # Извлечение zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    # Извлечение спектральных коэффициентов
    spectrogram = librosa.stft(y=audio)
    spectral_coefficients = librosa.amplitude_to_db(abs(spectrogram))
    # Продолжительность аудиофайла в секундах
    duration = librosa.get_duration(y=audio, sr=sr)
    # RMS (root mean square)
    rms = librosa.feature.rms(y=audio)

    return calculate_dist(mfcc.flatten()), calculate_dist(chroma.flatten()), calculate_dist(beat_frames.flatten()), calculate_dist(contrast.flatten()), calculate_dist(zcr.flatten()), calculate_dist(spectral_coefficients.flatten()), duration, calculate_dist(rms.flatten())



# Метрики
def are_elements_in_same_row(matrix, element1, element2):
    for row in matrix:
        if element1 in row and element2 in row:
            return True
    return False


def get_clusters(example_labels, calculated_labels):
    max_example_label = max(label for label in example_labels)
    max_calculated_label = max(label for label in calculated_labels)
    example_clusters = [[] for i in range(max_example_label+1)]
    calculated_clusters = [[] for i in range(max_calculated_label+1)]
    for i, label in enumerate(example_labels):
        if label != -1:
            example_clusters[label].append(i)
    for i, label in enumerate(calculated_labels):
        if label != -1:
            calculated_clusters[label].append(i)
    return example_clusters, calculated_clusters


def classification_metric(example_labels, calculated_labels):
    example_clusters, calculated_clusters = get_clusters(example_labels, calculated_labels)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(example_labels)):
        for j in range(len(example_labels)):
            in_same_example_cluster = are_elements_in_same_row(example_clusters, i, j)
            in_same_calculated_cluster = are_elements_in_same_row(calculated_clusters, i, j)
            if in_same_example_cluster and in_same_calculated_cluster: TP+=1
            elif not in_same_example_cluster and in_same_calculated_cluster: FP+=1
            elif not in_same_example_cluster and not in_same_calculated_cluster: TN+=1
            else: FN+=1
    if TP == 0: return 0
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F = 2 * P * R / (P + R)
    return F


def comparation_of_sets_metric(example_labels, calculated_labels):
    example_clusters, calculated_clusters = get_clusters(example_labels, calculated_labels)
    F_matrix = [[] for i in range(len(example_clusters))]
    for i, example_cluster in enumerate(example_clusters):
        for calculated_cluster in calculated_clusters:
            intersection = set(example_cluster) & set(calculated_cluster)
            P = len(intersection) / len (calculated_cluster)
            R = len(intersection) / len (example_cluster)
            F = 0
            if not P + R == 0:
                F = 2 * P * R / (P + R)
            F_matrix[i].append(F)
    F = 0
    for j in range(len(example_clusters)):
        F += len(example_clusters[j]) / len(example_clusters) * (max(F_value for F_value in F_matrix[j]) if F_matrix[j] else 0)
    return F




def get_file_type(file):
    # создаётся объект класса Magic, с помощью него читается информация о типе файла в виде тип/расширение (например 'text/plain', 'image/jpeg', 'audio/mp3')
    return magic.Magic(mime=True, magic_file="magic.mgc").from_file(file).split("/")


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def multi_clust(features_lists, num_of_clust):
    features_clusteraised_kmeans = []
    features_clusteraised_dbscan = []
    features_clusteraised_meanshift = []
    for feature_list in features_lists:
        if is_iterable(feature_list[0]):
            kmeans = KMeans(n_clusters=num_of_clust, n_init=10)
            dbscan = DBSCAN(eps=0.5, min_samples=2)
            meanshift = MeanShift(bandwidth=0.5)
            kmeans.fit(feature_list)
            dbscan.fit(feature_list)
            meanshift.fit(feature_list)
            features_clusteraised_kmeans.append(kmeans.labels_)
            features_clusteraised_dbscan.append(dbscan.labels_)
            features_clusteraised_meanshift.append(meanshift.labels_)
        else:
            features_clusteraised_kmeans.append(feature_list)
            features_clusteraised_dbscan.append(feature_list)
            features_clusteraised_meanshift.append(feature_list)
    features_clusteraised_kmeans = [[raw[i] for raw in features_clusteraised_kmeans] for i in range(len(features_clusteraised_kmeans[0]))]
    features_clusteraised_dbscan = [[raw[i] for raw in features_clusteraised_dbscan] for i in range(len(features_clusteraised_dbscan[0]))]
    features_clusteraised_meanshift = [[raw[i] for raw in features_clusteraised_meanshift] for i in range(len(features_clusteraised_meanshift[0]))]
    kmeans = KMeans(n_clusters=num_of_clust, n_init=10)
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    meanshift = MeanShift(bandwidth=0.5)
    kmeans.fit(features_clusteraised_kmeans)
    dbscan.fit(features_clusteraised_dbscan)
    meanshift.fit(features_clusteraised_meanshift)
    return kmeans.labels_, dbscan.labels_, meanshift.labels_



if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Запускать так:", sys.argv[0], "<директория с файлами> <количество кластеров>")
        exit()

    directory = sys.argv[1]
    num_of_clust = int(sys.argv[2])
    file_names = os.listdir(directory)
    
    text_files = []
    texts_preprocessed = []

    image_files = []
    images_color_histograms = []
    images_hog_features = []
    images_texture_features = []
    images_contours = []

    audio_files = []
    audio_mfccs = []
    audio_chromas = []
    audio_tempos = []
    audio_beat_frames = []
    audio_contrast = []
    audio_zcr = []
    audio_spectral_coefficients = []
    audio_duration = []
    audio_mean_amplitude = []
    audio_energy = []
    audio_rms = []


    for file_name in file_names:
        print("Начата обработка файла", file_name)
        file_path = os.path.join(directory, file_name)
        file_type, _ = get_file_type(file_path)
        match file_type:
            case "text":
                text_files.append(file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    processed_text = preprocess_text(text)
                    texts_preprocessed.append(processed_text)
            case "image":
                image_files.append(file_name)
                image = cv2.imread(file_path)
                images_color_histograms.append(get_color_histogram(file_path))
                images_hog_features.append(calculate_hog_features(image))
                images_texture_features.append(calculate_texture_features(image))
                images_contours.append(get_image_contours(image))
            case "audio":
                audio_files.append(file_name)
                for features_list, feature in zip([audio_mfccs, audio_chromas, audio_beat_frames, audio_contrast, audio_zcr, audio_spectral_coefficients, audio_duration, audio_rms], get_audio_features(file_path)):
                    features_list.append(feature)
            case _:
                print(file_type)
        print("Файл", file_name, "обработан")
    print("\n")    
    
    # Вычисление TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts_preprocessed)

    # Кластеризация текста
    kmeans_text = KMeans(n_clusters=num_of_clust, n_init=10)
    dbscan_text = DBSCAN(eps=0.5, min_samples=2)
    meanshift_text = MeanShift(bandwidth=0.5)
    kmeans_text.fit(tfidf_matrix)
    dbscan_text.fit(tfidf_matrix)
    meanshift_text.fit(tfidf_matrix.toarray())
    text_labels_kmeans = kmeans_text.labels_
    text_labels_dbscan = dbscan_text.labels_
    text_labels_meanshift = meanshift_text.labels_
    for file, label_kmeans, label_dbscan, label_meanshift in zip(text_files, text_labels_kmeans, text_labels_dbscan, text_labels_meanshift):
        print(file, label_kmeans, label_dbscan, label_meanshift)
    print("\n")

    # Кластеризация картинок
    image_labels_kmeans, image_labels_dbscan, image_labels_meanshift = multi_clust([images_color_histograms, images_hog_features, images_texture_features, images_contours], num_of_clust)
    for file, label_kmeans, label_dbscan, label_meanshift in zip(image_files, image_labels_kmeans, image_labels_dbscan, image_labels_meanshift):
        print(file, label_kmeans, label_dbscan, label_meanshift)
    print("\n")

    # Кластеризация аудио
    audio_labels_kmeans, audio_labels_dbscan, audio_labels_meanshift = multi_clust([audio_mfccs, audio_chromas, audio_beat_frames, audio_contrast, audio_zcr, audio_spectral_coefficients, audio_duration, audio_rms], num_of_clust)
    for file, label_kmeans, label_dbscan, label_meanshift in zip(audio_files, audio_labels_kmeans, audio_labels_dbscan, audio_labels_meanshift):
        print(file, label_kmeans, label_dbscan, label_meanshift)
    print("\n")

    # Оценка кластеризации
    example_labels = [0,0,0,1,1,1,2,2,2]
    print("Текст:")
    print("Подсчёт пар", classification_metric(example_labels, text_labels_kmeans), classification_metric(example_labels, text_labels_dbscan), classification_metric(example_labels, text_labels_meanshift))
    print("Сопоставление множеств", comparation_of_sets_metric(example_labels, text_labels_kmeans), comparation_of_sets_metric(example_labels, text_labels_dbscan), comparation_of_sets_metric(example_labels, text_labels_meanshift))
    print("Изображение:")
    print("Подсчёт пар", classification_metric(example_labels, image_labels_kmeans), classification_metric(example_labels, image_labels_dbscan), classification_metric(example_labels, image_labels_meanshift))
    print("Сопоставление множеств", comparation_of_sets_metric(example_labels, image_labels_kmeans), comparation_of_sets_metric(example_labels, image_labels_dbscan), comparation_of_sets_metric(example_labels, image_labels_meanshift))
    print("Аудио:")
    print("Подсчёт пар", classification_metric(example_labels, audio_labels_kmeans), classification_metric(example_labels, audio_labels_dbscan), classification_metric(example_labels, audio_labels_meanshift))
    print("Сопоставление множеств", comparation_of_sets_metric(example_labels, audio_labels_kmeans), comparation_of_sets_metric(example_labels, audio_labels_dbscan), comparation_of_sets_metric(example_labels, audio_labels_meanshift))