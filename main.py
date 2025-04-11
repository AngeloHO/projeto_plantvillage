from google.colab import drive
drive.mount('/content/drive')


# !pip install tensorflow
# !pip install matplotlib
# !pip install numpy


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

#caminho para o diretório do seu dataset
dataset_dir = '/content/drive/MyDrive/plantvillage dataset/'

# Definindo as pastas de treino e validação
train_dir = os.path.join(dataset_dir, 'color')
val_dir = os.path.join(dataset_dir, 'grayscale')

# trabalhar apenas com um tipo de imagem ex 80/20
# como dividir 80/20 as imagens ce treinamento e validação da mesma pasta
#cuidar para nao ter as mesmas imagens em validação e treino

# Verificando se as pastas existem
if not os.path.exists(train_dir):
    print(f"A pasta {train_dir} não existe.")
if not os.path.exists(val_dir):
    print(f"A pasta {val_dir} não existe.")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.5, 1.5],  # Ajuste de brilho
    channel_shift_range=20.0     # Mudança no canal de cor
)#data aumentation

val_datagen = ImageDataGenerator(rescale=1./255)  # Apenas normalização para o conjunto de validação
# Carregar as imagens de treino e validação com a nova resolução
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),        # Aumento da resolução para 224x224
    batch_size=32,                 #32 imagens por vezes(verificar quantas imagens podem ser carregadas pelo colab)
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),        # Aumento da resolução para 224x224
    batch_size=32,
    class_mode='categorical'
)
# Definindo um modelo baseado em ResNet50 pré-treinado
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#considerar outro modelo alem do resnet 50
# Congelar as camadas do modelo base para evitar o treinamento delas
base_model.trainable = False #se não ficar bom posso mudar para true e ver como se comporta

# Adicionando as camadas de classificação no topo
model = models.Sequential([
    base_model,  # Base da rede neural pré-treinada
    layers.GlobalAveragePooling2D(),  # Pooling global para reduzir as dimensões
    layers.Dense(512, activation='relu'),  # Camada densa adicional
    layers.Dropout(0.5),  # Dropout para evitar overfitting
    layers.Dense(38, activation='softmax')  # Camada de saída com 38 classes
])

# Compilando o modelo com um otimizador adaptativo
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Definindo os callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model_resnet50.h5', monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


steps_per_epoch = 300

# Definir o número de épocas
epochs = 100

# Early stopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Salvar o melhor modelo baseado na precisão de validação
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

# Reduzir a taxa de aprendizado se a perda de validação não melhorar
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

# Treinamento do modelo com os callbacks ajustados
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Processar todas as imagens por época
    epochs=epochs,  # Treinar por 20 épocas
    validation_data=validation_generator,
    validation_steps=10,  # Número de batches para validação
    callbacks=[early_stopping, checkpoint, reduce_lr]  # Usando os callbacks
)
