from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Carregar dados do CIFAR-10
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalizar os valores dos pixels para [0, 1]
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Achatar labels
train_labels = train_labels.ravel()
test_labels = test_labels.ravel()

# Tamanho do dataset
subset_size = 1000  # Ajuste para testes iniciais
train_images, train_labels = train_images[:subset_size], train_labels[:subset_size]
test_images, test_labels = test_images[:int(subset_size * 0.2)], test_labels[:int(subset_size * 0.2)]

# Definir nomes das classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Função para redimensionamento e upscaling das imagens
def preprocess_and_resize(image, label, img_size=224):
    image = tf.image.resize(image, (img_size, img_size))  # Redimensionar para 224x224
    return image, label

# Função de upscaling
def upscale_images(image):
    # Aplicando um simples upscaling com interpolação bilinear
    return tf.image.resize(image, (224, 224))

# Usar MobileNetV2 pré-treinada
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congelar as camadas do modelo base

# Modelo customizado
def create_model():
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')  # CIFAR-10 tem 10 classes
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Criar datasets com redimensionamento (sem upscaling)
train_dataset_no_upscale = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset_no_upscale = train_dataset_no_upscale.map(lambda x, y: preprocess_and_resize(x, y, img_size=224)).batch(64).shuffle(1000)

test_dataset_no_upscale = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset_no_upscale = test_dataset_no_upscale.map(lambda x, y: preprocess_and_resize(x, y, img_size=224)).batch(64)

# Treinar o modelo sem upscaling
model_no_upscale = create_model()
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history_no_upscale = model_no_upscale.fit(
    train_dataset_no_upscale,
    epochs=10,
    validation_data=test_dataset_no_upscale,
    callbacks=[early_stopping]
)

# Avaliação no dataset de teste (sem upscaling)
test_loss_no_upscale, test_acc_no_upscale = model_no_upscale.evaluate(test_dataset_no_upscale, verbose=2)
print(f"Acurácia sem upscaling: {test_acc_no_upscale:.4f}")

# Criar datasets com upscaling
train_dataset_upscale = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset_upscale = train_dataset_upscale.map(lambda x, y: preprocess_and_resize(upscale_images(x), y)).batch(64).shuffle(1000)

test_dataset_upscale = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset_upscale = test_dataset_upscale.map(lambda x, y: preprocess_and_resize(upscale_images(x), y)).batch(64)

# Treinar o modelo com upscaling
model_upscale = create_model()

history_upscale = model_upscale.fit(
    train_dataset_upscale,
    epochs=10,
    validation_data=test_dataset_upscale,
    callbacks=[early_stopping]
)

# Avaliação no dataset de teste (com upscaling)
test_loss_upscale, test_acc_upscale = model_upscale.evaluate(test_dataset_upscale, verbose=2)
print(f"Acurácia com upscaling: {test_acc_upscale:.4f}")

# Comparação dos resultados
print(f"\nComparação de Acurácias:\n- Acurácia sem upscaling: {test_acc_no_upscale:.4f}\n- Acurácia com upscaling: {test_acc_upscale:.4f}")

# Exibição de uma imagem de inferência (pode ser do dataset de teste)
sample_image = test_images[0]  # Selecionando uma imagem aleatória do conjunto de teste
sample_image_resized = tf.image.resize(sample_image, (224, 224))  # Redimensionar para 224x224
sample_image_resized = np.expand_dims(sample_image_resized, axis=0)  # Adicionar a dimensão do batch

# Realizar inferência no modelo sem upscaling
prediction_no_upscale = model_no_upscale.predict(sample_image_resized)
predicted_class_no_upscale = class_names[np.argmax(prediction_no_upscale)]  # Obter a classe prevista
print(f"Classe prevista para o modelo sem upscaling: {predicted_class_no_upscale}")

# Realizar inferência no modelo com upscaling
sample_image_upscaled = upscale_images(sample_image)  # Upscale da imagem
sample_image_upscaled = np.expand_dims(sample_image_upscaled, axis=0)  # Adicionar a dimensão do batch
prediction_upscale = model_upscale.predict(sample_image_upscaled)
predicted_class_upscale = class_names[np.argmax(prediction_upscale)]  # Obter a classe prevista
print(f"Classe prevista para o modelo com upscaling: {predicted_class_upscale}")

# Matriz de confusão para comparação (sem upscaling)
predictions_no_upscale = np.argmax(model_no_upscale.predict(test_dataset_no_upscale), axis=1)
cm_no_upscale = confusion_matrix(test_labels[:len(predictions_no_upscale)], predictions_no_upscale)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_no_upscale, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão - Sem Upscaling')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Real')
plt.show()

# Matriz de confusão para comparação (com upscaling)
predictions_upscale = np.argmax(model_upscale.predict(test_dataset_upscale), axis=1)
cm_upscale = confusion_matrix(test_labels[:len(predictions_upscale)], predictions_upscale)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_upscale, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão - Com Upscaling')
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Real')
plt.show()