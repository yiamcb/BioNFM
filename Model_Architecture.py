import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_classes):
    """
    Create a biologically informed deep learning model for EEG data with input shape (40, 5).

    Parameters:
    - input_shape: Tuple, shape of the input EEG data.
    - num_classes: Integer, number of output classes.

    Returns:
    - model: Compiled Keras model.
    """
    inputs = layers.Input(shape=input_shape)  # (40, 5)

    ### Graph Neural Network Module ###
    # Simulating adjacency-like learning through Dense layer interaction
    adjacency_encoding = layers.Dense(input_shape[1], activation='relu')(inputs)  # (40, 5)
    gnn_output = layers.Dense(64, activation='relu')(adjacency_encoding)  # (40, 64)

    ### Transformer for Temporal Dynamics ###
    # Self-attention mechanism
    x = inputs  # Shape (40, 5)
    transformer_output = layers.MultiHeadAttention(num_heads=4, key_dim=5)(x, x)  # (40, 5)
    transformer_output = layers.LayerNormalization(epsilon=1e-6)(transformer_output + x)

    # Feed-forward network within the transformer
    ff_layer = layers.Dense(64, activation="relu")(transformer_output)  # (40, 64)
    ff_layer = layers.Dense(input_shape[1], activation="relu")(ff_layer)  # Project back to (40, 5)
    x = layers.LayerNormalization(epsilon=1e-6)(ff_layer + transformer_output)  # Residual connection (40, 5)

    ### Reservoir Computing Module (Simulated) ###
    # Adding non-linear temporal dynamics
    reservoir_output = layers.SimpleRNN(128, return_sequences=False, activation="tanh")(x)  # (128)

    ### Cross-Frequency Coupling Integration ###
    # Simulating biologically inspired coupling features as additional input
    coupling_features = layers.Dense(32, activation="relu")(inputs)  # (40, 32)
    coupling_flattened = layers.Flatten()(coupling_features)  # (40 * 32)

    ### Merge Features ###
    gnn_flattened = layers.Flatten()(gnn_output)  # (40 * 64)
    merged = layers.concatenate([gnn_flattened, reservoir_output, coupling_flattened])  # Combined features

    ### Fully Connected Layers ###
    x = layers.Dense(128, activation="relu")(merged)  # (128)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)  # Output layer

    model = models.Model(inputs, outputs)
    return model


num_classes = encoded_labels.shape[1]
input_shape = X_train.shape[1:]  # (40, 5)
model = create_model(input_shape, num_classes)

# Custom Loss Function: Incorporate Biological Constraints
def custom_loss_function(y_true, y_pred):
    # Cross-entropy loss
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Regularization: Encourage biologically plausible predictions
    beta_coherence_penalty = tf.reduce_mean(tf.abs(y_pred[:, 0] - y_pred[:, 1]))  # Example penalty
    entropy_penalty = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-8))

    # Combine losses
    loss = cross_entropy + 0.01 * beta_coherence_penalty + 0.01 * entropy_penalty
    return loss

model.compile(optimizer="adam", loss=custom_loss_function, metrics=["accuracy"])

model.summary()