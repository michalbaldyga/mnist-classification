import tensorflow as tf
import tensorflow_datasets as tfds

# Load MNIST dataset
dataset, dataset_info = tfds.load(name='mnist', as_supervised=True, with_info=True)

# Split dataset to train_set and test_set
raw_train_data, raw_test_data = dataset['train'], dataset['test']

# Scale data
scaled_train_data = raw_train_data.map(lambda image, label: (tf.cast(image, tf.float32) / 255., label))
test_data = raw_test_data.map(lambda image, label: (tf.cast(image, tf.float32) / 255., label))

# Shuffle train_set
BUFFER_SIZE = 1000
shuffled_train_data = scaled_train_data.shuffle(BUFFER_SIZE)

# Split train_set to train_set and validation_set (90/10)
validation_samples_no = 0.1 * dataset_info.splits['train'].num_examples
validation_samples_no = tf.cast(validation_samples_no, tf.int64)
validation_data = shuffled_train_data.take(validation_samples_no)
train_data = shuffled_train_data.skip(validation_samples_no)

# Split data to batches
BATCH_SIZE = 50
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(validation_samples_no)
validation_inputs, validation_targets = next(iter(validation_data))

test_samples_no = dataset_info.splits['test'].num_examples
test_samples_no = tf.cast(test_samples_no, tf.int64)
test_data = test_data.batch(test_samples_no)

# Create the model
OUTPUT_SIZE = 10
HIDDEN_LAYER_SIZE = 200

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'),
        tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu'),
        tf.keras.layers.Dense(OUTPUT_SIZE, activation='softmax')
    ]
)

# Select the Loss and the Optimizer
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=custom_optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_data,
    epochs=5,
    validation_data=(validation_inputs, validation_targets),
    verbose=2)

# Test the model
test_loss, test_accuracy = model.evaluate(test_data)
print('Test loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))
