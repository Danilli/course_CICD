import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
import numpy as np

# Папка с моделями автоенкодеров
base_folder = 'besties_party'
models = {}

# Получаем список всех папок в базовой папке
model_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f)) and f != '.ipynb_checkpoints']

# Загрузка моделей
for model_folder in model_folders:
    model_path = os.path.join(base_folder, model_folder)
    memAE_file_path = os.path.join(model_path, 'trained_model.keras')
    models[model_folder] = load_model(memAE_file_path)

# Загрузка данных
data = pd.read_csv('../data/scaled_data.csv', index_col="kkt_id")

# Вывод информации о моделях
for model_name, model in models.items():
    print(model_name)
    model.summary()

# Папка для сохранения разделенных моделей
output_folder = 'AE_Split'
os.makedirs(output_folder, exist_ok=True)

# Функция для разделения модели на энкодер и декодер
def split_model(model, model_name, is_memae=False):
    # Определяем количество слоев для декодера
    if is_memae:
        decoder_layers = model.layers[-6:]
    else:
        decoder_layers = model.layers[-4:]
    
    # Создаем энкодер
    encoder_input = Input(shape=model.input_shape[1:])
    encoder_output = encoder_input
    for layer in model.layers[1:-len(decoder_layers)]:  # Пропускаем InputLayer
        if isinstance(layer, Model):
            encoder_output = layer(encoder_output)
        else:
            encoder_output = layer(encoder_output)
    encoder = Model(encoder_input, encoder_output, name=f'{model_name}_encoder')
    
    # Создаем декодер
    decoder_input = Input(shape=encoder.output_shape[1:])
    decoder_output = decoder_input
    for layer in decoder_layers:
        if isinstance(layer, Model):
            decoder_output = layer(decoder_output)
        else:
            decoder_output = layer(decoder_output)
    decoder = Model(decoder_input, decoder_output, name=f'{model_name}_decoder')
    
    # Сохраняем модели
    encoder.save(os.path.join(output_folder, f'{model_name}_encoder.keras'))
    decoder.save(os.path.join(output_folder, f'{model_name}_decoder.keras'))
    
    return encoder, decoder

# Разделяем каждую модель на энкодер и декодер
encoders = {}
decoders = {}

for model_name, model in models.items():
    encoder, decoder = split_model(model, model_name)
    encoders[model_name] = encoder
    decoders[model_name] = decoder

# Словарь для хранения эмбеддингов
embeddings_dict = {}

# Генерация эмбеддингов для каждой модели
for model_name, model in encoders.items():
    # Преобразование данных в numpy массив
    data_array = data.values
    
    # Генерация эмбеддингов
    embeddings = model.predict(data_array)
    
    # Создание DataFrame с эмбеддингами
    embeddings_df = pd.DataFrame(embeddings,
                                 columns=[f'embedding_{i}' for i in range(embeddings.shape[1])],
                                 index=data.index)
    
    # Сохранение эмбеддингов в словарь
    embeddings_dict[model_name] = embeddings_df

# Вывод статистики по эмбеддингам
for model_name, embeddings_df in embeddings_dict.items():
    print(f"Embeddings for model: {model_name}")
    print(embeddings_df.describe())

# Сохранение эмбеддингов в CSV файлы
for model_name, embeddings_df in embeddings_dict.items():
    embeddings_df.to_csv(f'../data/embeddings/{model_name}_embeddings.csv', index=True)