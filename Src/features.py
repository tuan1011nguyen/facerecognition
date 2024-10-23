import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from data_loader import load_image
from model import vgg_face
from tensorflow.keras.models import Model
import pickle

current_dir = os.path.dirname(__file__)  # Thư mục hiện tại chứa file Python này

embeddings_file = os.path.join(current_dir, '../model/embeddings.pkl')
weight_path = os.path.join(current_dir, '../model/vgg_face_weights.h5')
metadata_file = os.path.join(current_dir, '../model/metadata.pkl')
embeddings_file = os.path.join(current_dir, '../model/embeddings.pkl')


def compute_embeddings(metadata):
    model = vgg_face()
    model.load_weights(weight_path)
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    metadata_array = np.array(metadata)
    
    embeddings = np.zeros((metadata_array.shape[0], 2622))
    for i, m in enumerate(metadata):
        img_path = m.image_path()
        img = load_image(img_path)
        img = (img / 255.).astype(np.float32)
        img = cv2.resize(img, dsize=(224, 224))
        embedding_vector = vgg_face_descriptor.predict(np.expand_dims(img, axis=0))[0]
        embeddings[i] = embedding_vector

    return embeddings

def compute_embedding_for_image(img):
    model = vgg_face()
    model.load_weights(weight_path)
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    
    img = (img / 255.).astype(np.float32)  # Chuẩn hóa ảnh
    img = cv2.resize(img, dsize=(224, 224))  # Thay đổi kích thước ảnh
    embedding_vector = vgg_face_descriptor.predict(np.expand_dims(img, axis=0))[0]  # Tính toán embedding
    return embedding_vector


def save_embeddings_and_metadata(embeddings, metadata, embeddings_file=embeddings_file, metadata_file=metadata_file):
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

def load_embeddings_and_metadata(embeddings_file=embeddings_file, metadata_file=metadata_file):
    with open(embeddings_file, 'rb') as f:
        embeddings = pickle.load(f)

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    return embeddings, metadata

