import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, LeakyReLU, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
import statistics


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (root.find('filename').text, 
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),                      
                     member.find('name').text,
                     int(bndbox.find('xmin').text),                     
                     int(bndbox.find('ymin').text),                     
                     int(bndbox.find('xmax').text),                     
                     int(bndbox.find('ymax').text))  
            # print(value)          
            xml_list.append(value)    
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


# Process images and extract features
def generate_images(fin_images):
    source_images = fin_images['filename'].to_list()
    images = []
    for i in source_images:
        path = './dataset/train/' + i
        input_image = cv2.imread(path)
        input_image = cv2.resize(input_image, (255, 255))
        images.append(input_image)
    return np.array(images), source_images


def generate_keypoints(fin_images, source_images):
    keypoint_features = []
    for i in source_images:
        image_name = i
        mask = fin_images[fin_images['filename'] == image_name]
        single_image_data = mask.iloc[0]
        # print(single_image_data)
        width_ratio = single_image_data['width'] / 255
        height_ratio = single_image_data['height'] / 255

        keypoints = [
            single_image_data['xmin'] / width_ratio,
            single_image_data['ymin'] / height_ratio,
            single_image_data['xmax'] / width_ratio,
            single_image_data['ymax'] / height_ratio
        ]

        # print(keypoints)
        # exit()

        keypoint_features.append(keypoints)
    
    
    return np.array(keypoint_features, dtype=float)


# Define the neural network model
def build_model(input_shape, no_of_keypoints):
    model = Sequential()
    
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=input_shape))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    
    # Input dimensions: (None, 96, 96, 32)
    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    # Input dimensions: (None, 48, 48, 32)
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    
    # Input dimensions: (None, 48, 48, 64)
    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
   
    # Input dimensions: (None, 24, 24, 64)
    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    
    # Input dimensions: (None, 24, 24, 96)
    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    # Input dimensions: (None, 12, 12, 96)
    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    
    # Input dimensions: (None, 12, 12, 128)
    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
   
    # Input dimensions: (None, 6, 6, 128)
    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    
    # Input dimensions: (None, 6, 6, 256)
    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    # Input dimensions: (None, 3, 3, 256)
    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    
    # Input dimensions: (None, 3, 3, 512)
    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    
    # Input dimensions: (None, 3, 3, 512)
    model.add(Flatten())
    model.add(Dense(512, activation='linear'))
    model.add(Dropout(0.5))
    model.add(Dense(no_of_keypoints))
    return model


def build_model_from_trained_model(input_shape, no_of_keypoints):
    base_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Conv2D(32, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation='linear')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(no_of_keypoints, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model
    
def main():
    image_path = os.path.join(os.getcwd(), './dataset/train')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('labels.csv', index=None)

    req_images = pd.read_csv('labels.csv')
    fin_images = req_images.drop_duplicates(subset='filename', keep="first")
    model_input_images, source_images = generate_images(fin_images)


    model_input_keypoints = generate_keypoints(fin_images, source_images)

    input_shape = (255, 255, 3)
    no_of_keypoints = 4
    model = build_model(input_shape, no_of_keypoints)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

    earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min',restore_best_weights=True)
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=1e-15, mode='min', verbose=1)
    history = model.fit(model_input_images, model_input_keypoints, epochs=200, batch_size=8, validation_split=0.2, callbacks=[earlyStopping, rlp])

    plt.plot(history.history['loss'][10:])  
    plt.plot(history.history['val_loss'][10:])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("./Loss-Graph.png")
    plt.close()
    # plt.show()
    pattern = 'battle-tank-detector*.h5'

    existing_files = glob.glob(pattern)
    file_count = len(existing_files)
    if file_count > 0:
        new_file_name = f'battle-tank-detector-{file_count + 1}.h5'
    else:
        new_file_name = 'battle-tank-detector-1.h5'

    model.save(new_file_name)
    test_model(model, './Abrams_in_formation.jpg')

    
def test_model(model, test_image_path):
    test_image = cv2.imread(test_image_path)
    original_size = test_image.shape[:2]  # height, width
    test_image_resized = cv2.resize(test_image, (255, 255))
    test_image_normalized = test_image_resized / 255.0

    test_input = np.expand_dims(test_image_normalized, axis=0)
    predictedResults = model.predict(test_input)
    print(predictedResults)
    predicted_keypoints = (predictedResults[0]).astype(int)

    output_image = cv2.rectangle(test_image_resized, 
                                 (predicted_keypoints[0], predicted_keypoints[1]), 
                                 (predicted_keypoints[2], predicted_keypoints[3]), 
                                 (255, 0, 0), 2)

    plt.figure(figsize=(12, 5))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.savefig("prediction-223.png")
    plt.show()



# main()

from keras.models import load_model
loaded_model = load_model("./battle-tank-detector-5.h5")
test_model(loaded_model, './fig3.png')