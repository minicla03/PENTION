import tensorflow as tf
from tensorflow.keras import layers, models, initializers
import numpy as np

class GraphConvLayer(layers.Layer):
    """Custom Graph Convolution Layer"""
    def __init__(self, units, activation='relu', **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        # input_shape[0] = (batch, nodes, features)
        # input_shape[1] = (batch, nodes, nodes) adjacency matrix
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel'
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        super(GraphConvLayer, self).build(input_shape)
    
    def call(self, inputs):
        features, adjacency = inputs
        
        # Graph convolution: A * X * W + b
        # features: (batch, nodes, input_features)
        # adjacency: (batch, nodes, nodes)
        
        # Linear transformation: X * W
        output = tf.matmul(features, self.kernel)
        
        # Graph convolution: A * (X * W)
        output = tf.matmul(adjacency, output)
        
        # Add bias
        output = tf.nn.bias_add(output, self.bias)
        
        # Apply activation
        if self.activation is not None:
            output = self.activation(output)
            
        return output

def build_graph_autoencoder(num_nodes=100, input_features=10, hidden_dims=[64, 32, 16]):
    """
    Costruisce un Graph Autoencoder
    
    Args:
        num_nodes: numero di nodi nel grafo
        input_features: dimensione features per ogni nodo
        hidden_dims: dimensioni dei layer nascosti
    """
    
    # Input layers
    node_features = layers.Input(shape=(num_nodes, input_features), name="node_features")
    adjacency_matrix = layers.Input(shape=(num_nodes, num_nodes), name="adjacency_matrix")
    
    # Normalizzazione della matrice di adiacenza (aggiunge self-loops)
    adj_normalized = layers.Lambda(
        lambda x: x + tf.eye(tf.shape(x)[1], batch_shape=[tf.shape(x)[0]]),
        name="add_self_loops"
    )(adjacency_matrix)
    
    # Degree normalization
    adj_normalized = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=-1),
        name="normalize_adjacency"
    )(adj_normalized)
    
    # ENCODER
    x = node_features
    encoder_layers = []
    
    for i, hidden_dim in enumerate(hidden_dims):
        # Graph convolution
        x = GraphConvLayer(
            units=hidden_dim, 
            activation='relu', 
            name=f"encoder_gcn_{i+1}"
        )([x, adj_normalized])
        
        # Batch normalization
        x = layers.BatchNormalization(name=f"encoder_bn_{i+1}")(x)
        
        # Dropout per regolarizzazione
        x = layers.Dropout(0.1, name=f"encoder_dropout_{i+1}")(x)
        
        encoder_layers.append(x)
    
    # Rappresentazione latente (encoded)
    encoded = x
    
    # DECODER
    # Ricostruiamo dalle dimensioni nascoste fino alle features originali
    decoder_dims = hidden_dims[:-1][::-1] + [input_features]  # reverse + original dim
    
    x = encoded
    for i, output_dim in enumerate(decoder_dims):
        activation = 'relu' if i < len(decoder_dims) - 1 else None  # No activation on last layer
        
        x = GraphConvLayer(
            units=output_dim, 
            activation=activation, 
            name=f"decoder_gcn_{i+1}"
        )([x, adj_normalized])
        
        if i < len(decoder_dims) - 1:  # No BatchNorm on output layer
            x = layers.BatchNormalization(name=f"decoder_bn_{i+1}")(x)
            x = layers.Dropout(0.1, name=f"decoder_dropout_{i+1}")(x)
    
    # Output ricostruito
    decoded = x
    
    # Modello completo
    autoencoder = models.Model(
        inputs=[node_features, adjacency_matrix], 
        outputs=decoded, 
        name="GraphAutoencoder"
    )
    
    return autoencoder

# Esempio di utilizzo
def create_sample_graph_data(batch_size=32, num_nodes=50, input_features=10):
    """Crea dati di esempio per testare la GNN"""
    
    # Features casuali per i nodi
    node_features = np.random.randn(batch_size, num_nodes, input_features)
    
    # Matrice di adiacenza casuale (grafo sparso)
    adjacency = np.random.rand(batch_size, num_nodes, num_nodes)
    adjacency = (adjacency > 0.8).astype(np.float32)  # Grafo sparso
    
    # Rendi la matrice simmetrica (grafo non diretto)
    adjacency = (adjacency + np.transpose(adjacency, (0, 2, 1))) / 2
    adjacency = (adjacency > 0).astype(np.float32)
    
    return node_features, adjacency

# Creazione e test del modello
if __name__ == "__main__":
    # Parametri
    num_nodes = 50
    input_features = 10
    batch_size = 32
    
    # Crea il modello
    model = build_graph_autoencoder(
        num_nodes=num_nodes, 
        input_features=input_features,
        hidden_dims=[64, 32, 16]
    )
    
    # Compila il modello
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Crea dati di esempio
    X_nodes, X_adj = create_sample_graph_data(batch_size, num_nodes, input_features)
    
    # Summary del modello
    print("=== GRAPH AUTOENCODER SUMMARY ===")
    model.summary()
    
    # Test forward pass
    print(f"\nInput shape - Nodes: {X_nodes.shape}")
    print(f"Input shape - Adjacency: {X_adj.shape}")
    
    # Prediction
    reconstructed = model.predict([X_nodes, X_adj])
    print(f"Output shape: {reconstructed.shape}")
    
    # Train per pochi step come esempio
    print("\n=== TRAINING EXAMPLE ===")
    history = model.fit(
        [X_nodes, X_adj], 
        X_nodes,  # Target = input (autoencoder)
        batch_size=8,
        epochs=3,
        verbose=1
    )
    
    print(f"Final loss: {history.history['loss'][-1]:.4f}")