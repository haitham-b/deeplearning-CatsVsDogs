# import keras packages
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing.image import *
from keras.applications.mobilenetv2 import MobileNetV2

# define some variables
DATASET_PATH = './data/sample'
FREEZE_LAYERS = 2  # freeze this many layers for training
IMAGE_SIZE = (224, 224)

NUM_CLASSES = 2
CLASS_LABEL = ["cat", "dog"]

DATASET_PATH = './data/sample'
BATCH_SIZE = 8  # try reducing batch size or freeze more layers if your CPU or GPU runs out of memory
NUM_EPOCHS = 2

file_checkpoints = "MobileNetV2-weights-improvement.hdf5"
file_weights = "model-MobileNetV2-final.h5"
file_architecture = 'model_MobileNetV2_architecture.json'


def main():
    """
    Script entrypoint
    """
    train_data = ImageDataGenerator(rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    channel_shift_range=10,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

    train_batches = train_data.flow_from_directory(DATASET_PATH + '/train',
                                                   target_size=IMAGE_SIZE,
                                                   interpolation='bicubic',
                                                   class_mode='categorical',
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE)

    valid_data = ImageDataGenerator()
    valid_batches = valid_data.flow_from_directory(DATASET_PATH + '/valid',
                                                   target_size=IMAGE_SIZE,
                                                   interpolation='bicubic',
                                                   class_mode='categorical',
                                                   shuffle=True,
                                                   batch_size=BATCH_SIZE)

    dnn = MobileNetV2(include_top=False, input_tensor=None, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    final_dnn = prepare_model(dnn, NUM_CLASSES)
    train(network=final_dnn, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, training=train_batches, validation=valid_batches)

    # save trained weights
    final_dnn.save_weights(file_weights)
    with open(file_architecture, 'w') as f:
        f.write(final_dnn.to_json())


def train(network, epochs, batch_size, training, validation):
    # Use Checkpoint model to save model weights after an epoch
    checkpoint_callback = ModelCheckpoint(file_checkpoints, monitor='val_acc', verbose=1, save_best_only=True,
                                          mode='max')

    # Configure Tensorboard Callback
    tensorboard_callback = TensorBoard(log_dir='./tensorboard', histogram_freq=0, write_graph=True,
                                       write_images=False)

    network.fit_generator(training, steps_per_epoch=training.samples // batch_size,
                          validation_data=validation,
                          validation_steps=validation.samples // batch_size,
                          epochs=epochs,
                          callbacks=[checkpoint_callback, tensorboard_callback])


def prepare_model(network, num_classes):
    x = network.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)  # to prevent overfitting by randomly remove neruons during forward or backpropegation

    output_layer = Dense(num_classes, activation='softmax', name='softmax')(x)
    net_final = Model(inputs=network.input, outputs=output_layer)

    for layer in net_final.layers[:FREEZE_LAYERS]:
        layer.trainable = False
    for layer in net_final.layers[FREEZE_LAYERS:]:
        layer.trainable = True

    net_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    return net_final


if __name__ == "__main__":
    main()
