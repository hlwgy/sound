
# %% TensorFlow 2.6(2021年10之后版) + python 3.9
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
import matplotlib.pyplot as plt

# %% 组织数据
data_dir = "datasets" # 数据集文件夹
# 获取所有分类
labels = np.array(tf.io.gfile.listdir(str(data_dir)))
# labels # ['go', 'left', 'right', 'stop']
# 获取所有文件
filenames = tf.io.gfile.glob(data_dir + '/*/*.wav')
filenames = tf.random.shuffle(filenames) # 顺序打乱
# filenames # ['datasets\\go\\1996.wav',……]

# 根据文件路径，组装（音频文件输入，标签结果输出），用于训练
def get_audio_and_label(file_path):
    # 根据路径取标签 datasets\\no\\1.wav
    parts = tf.strings.split(input=file_path,sep=os.path.sep)
    label = parts[-2] # 取出 no
    # 获得位于数组中索引
    label_id = tf.math.argmax(label == labels) 
    # 音频文件处理成数据
    audio_data = get_audio_data(file_path)
    return audio_data, label_id

# 音频文件处理成数据
def get_audio_data(file_path):
    # 根据路径解码音频文件，时域
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    waveform = tf.squeeze(audio, axis=-1)

    # 根据时域，获取声谱，频域
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros([16000] - tf.shape(waveform),dtype=tf.float32)
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# 处理成TF格式数据集
def preprocess_dataset(filenames):

    batch_size = 64
    AUTOTUNE = tf.data.AUTOTUNE
    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
    files_ds = filenames_ds.map(map_func=get_audio_and_label, num_parallel_calls=AUTOTUNE)
    datasets = files_ds.batch(batch_size)
    datasets = datasets.cache().prefetch(AUTOTUNE)

    return files_ds, datasets

train_files_ds, train_ds = preprocess_dataset(filenames[:3000])
val_files_ds, val_ds = preprocess_dataset(filenames[3000:])
checkpoint_save_path = 'model/model.ckpt'  

# 创建模型
def create_model():
    model = models.Sequential([
        layers.Input(shape= (124, 129, 1)),
        layers.Resizing(32, 32),
        layers.Normalization(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    if os.path.exists(checkpoint_save_path + '.index'):
        model.load_weights(checkpoint_save_path)  

    return model

# %% 训练数据
model = create_model()
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,save_weights_only=True, save_best_only=True)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[cp_callback]
)

# 显示准确率随着训练次数的曲线
metrics = history.history
plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'])
plt.show()
# %% 验证数据
model = create_model()
labels = ['go', 'left', 'right', 'stop']
# 音频文件转码
audio = get_audio_data('datasets/go/2001.wav')
audios = np.array([audio])
predictions = model(audios)
index = np.argmax(predictions[0])
print(labels[index])