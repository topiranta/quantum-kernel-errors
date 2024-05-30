############################################
# Title: Utility Functions For Quantum Prediction Advantage Error Simulation
# Based on: https://www.tensorflow.org/quantum/tutorials/quantum_data and https://www.tensorflow.org/quantum/tutorials/error_simulation
# License: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0
############################################

import importlib, pkg_resources
importlib.reload(pkg_resources)

import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from cirq.contrib.svg import SVGCircuit
np.random.seed(1234)

#### Meta Methods

def get_file_location():
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    return __location__

def colorpalette():
    
    palette = {
        'orange': '#fca311',
        'charcoal': '#424b54',
        'white': '#ffffff',
        'mantis': '#79b473',
        'cambridge': '#70a37f',
        'carmine': '#e63946',
        'celadon': '#457b9d',
        'sandy': '#f4a261',
        'slate': '#8d99ae'
    }

    return palette

#### Data Processing Methods

def load_mnist_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train/255.0, x_test/255.0

    return x_train, x_test, y_train, y_test

def filter_03(x, y):
    keep = (y == 0) | (y == 3)
    x, y = x[keep], y[keep]
    y = y == 0
    return x,y

def truncate_x(x_train, x_test, n_components=10):
  """Perform PCA on image dataset keeping the top `n_components` components."""
  n_points_train = tf.gather(tf.shape(x_train), 0)
  n_points_test = tf.gather(tf.shape(x_test), 0)

  # Flatten to 1D
  x_train = tf.reshape(x_train, [n_points_train, -1])
  x_test = tf.reshape(x_test, [n_points_test, -1])

  # Normalize.
  feature_mean = tf.reduce_mean(x_train, axis=0)
  x_train_normalized = x_train - feature_mean
  x_test_normalized = x_test - feature_mean

  # Truncate.
  e_values, e_vectors = tf.linalg.eigh(
      tf.einsum('ji,jk->ik', x_train_normalized, x_train_normalized))
  return tf.einsum('ij,jk->ik', x_train_normalized, e_vectors[:,-n_components:]), \
    tf.einsum('ij,jk->ik', x_test_normalized, e_vectors[:, -n_components:])

def reduce(x_train, x_test, y_train, y_test, n_train, n_test):
   
  x_train_r, x_test_r = x_train[:n_train], x_test[:n_test]
  y_train_r, y_test_r = y_train[:n_train], y_test[:n_test]

  return x_train_r, x_test_r, y_train_r, y_test_r

def get_prepared_data(n_train, n_test, n_components=10):

  __location__ = get_file_location()
  try:
    x_train = np.load(os.path.join(__location__, f'data/x_train_{n_train}_d{n_components}.npy'), allow_pickle=True)
    x_test = np.load(os.path.join(__location__, f'data/x_test_{n_test}_d{n_components}.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(__location__, f'data/y_train_{n_train}_d{n_components}.npy'), allow_pickle=True)
    y_test = np.load(os.path.join(__location__, f'data/y_test_{n_test}_d{n_components}.npy'), allow_pickle=True)
  except:
    x_train, x_test, y_train, y_test = load_mnist_data()
    x_train, y_train = filter_03(x_train, y_train)
    x_test, y_test = filter_03(x_test, y_test)
    x_train, x_test = truncate_x(x_train, x_test, n_components)
    x_train, x_test, y_train, y_test = reduce(x_train, x_test, y_train, y_test, n_train, n_test)
    
    np.save(os.path.join(__location__, f'data/x_train_{n_train}_d{n_components}.npy'), x_train)
    np.save(os.path.join(__location__, f'data/x_test_{n_test}_d{n_components}.npy'), x_test)
    np.save(os.path.join(__location__, f'data/y_train_{n_train}_d{n_components}.npy'), y_train)
    np.save(os.path.join(__location__, f'data/y_test_{n_test}_d{n_components}.npy'), y_test)

  return x_train, x_test, y_train, y_test


#### Quantum Methods

def single_qubit_wall(qubits, rotations):
  """Prepare a single qubit X,Y,Z rotation wall on `qubits`."""
  wall_circuit = cirq.Circuit()
  for i, qubit in enumerate(qubits):
    for j, gate in enumerate([cirq.X, cirq.Y, cirq.Z]):
      wall_circuit.append(gate(qubit) ** rotations[i][j])

  return wall_circuit

def v_theta(qubits):
  """Prepares a circuit that generates V(\theta)."""
  ref_paulis = [
      cirq.X(q0) * cirq.X(q1) + \
      cirq.Y(q0) * cirq.Y(q1) + \
      cirq.Z(q0) * cirq.Z(q1) for q0, q1 in zip(qubits, qubits[1:])
  ]
  exp_symbols = list(sympy.symbols('ref_0:'+str(len(ref_paulis))))
  return tfq.util.exponential(ref_paulis, exp_symbols), exp_symbols

def prepare_pqk_circuits(qubits, classical_source, random_rots, p = 0.0, n_trotter=10):
  """Prepare the pqk feature circuits around a dataset."""
  n_qubits = len(qubits)
  n_points = len(classical_source)
  
  initial_U = single_qubit_wall(qubits, random_rots)

  if p >= 1e-5:
    initial_U = initial_U.with_noise(cirq.depolarize(p))

  # Prepare parametrized V
  V_circuit, symbols = v_theta(qubits)
  exp_circuit = cirq.Circuit(V_circuit for t in range(n_trotter))
  
  if p >= 1e-5:
    exp_circuit = exp_circuit.with_noise(cirq.depolarize(p))

  # Convert to `tf.Tensor`
  initial_U_tensor = tfq.convert_to_tensor([initial_U])
  initial_U_splat = tf.tile(initial_U_tensor, [n_points])

  full_circuits = tfq.layers.AddCircuit()(
      initial_U_splat, append=exp_circuit)
  # Replace placeholders in circuits with values from `classical_source`.
  return tfq.resolve_parameters(
      full_circuits, tf.convert_to_tensor([str(x) for x in symbols]),
      tf.convert_to_tensor(classical_source*(n_qubits/3)/n_trotter))

def get_pqk_features(qubits, data_batch, noisy=False, repetitions=1000):
  """Get PQK features based on above construction."""
  ops = [[cirq.X(q), cirq.Y(q), cirq.Z(q)] for q in qubits]
  ops_tensor = tf.expand_dims(tf.reshape(tfq.convert_to_tensor(ops), -1), 0)
  batch_dim = tf.gather(tf.shape(data_batch), 0)
  ops_splat = tf.tile(ops_tensor, [batch_dim, 1])
  if noisy:
    exp_vals = tfq.layers.SampledExpectation(backend='noisy')(data_batch, operators=ops_splat, repetitions=repetitions)
  else:
    exp_vals = tfq.layers.Expectation()(data_batch, operators=ops_splat)
  
  rdm = tf.reshape(exp_vals, [batch_dim, len(qubits), -1])
  return rdm

def compute_kernel_matrix(vecs, gamma):
  """Computes d[i][j] = e^ -gamma * (vecs[i] - vecs[j]) ** 2 """
  scaled_gamma = gamma / (
      tf.cast(tf.gather(tf.shape(vecs), 1), tf.float32) * tf.math.reduce_std(vecs))
  return scaled_gamma * tf.einsum('ijk->ij',(vecs[:,None,:] - vecs) ** 2)

def get_spectrum(datapoints, gamma=1.0):
  """Compute the eigenvalues and eigenvectors of the kernel of datapoints."""
  KC_qs = compute_kernel_matrix(datapoints, gamma)
  S, V = tf.linalg.eigh(KC_qs)
  S = tf.math.abs(S)
  return S, V

def get_stilted_dataset(S, V, S_2, V_2, lambdav=1.1):
  """Prepare new labels that maximize the classical model complexity."""
  S_diag = tf.linalg.diag(S ** 0.5)
  S_2_diag = tf.linalg.diag(S_2 / (S_2 + lambdav) ** 2)
  scaling = S_diag @ tf.transpose(V) @ \
            V_2 @ S_2_diag @ tf.transpose(V_2) @ \
            V @ S_diag

  # Generate new lables using the largest eigenvector.
  _, vecs = tf.linalg.eig(scaling)
  new_labels = tf.math.real(
      tf.einsum('ij,j->i', tf.cast(V @ S_diag, tf.complex64), vecs[-1])).numpy()
  # Create new labels and add some small amount of noise.
  final_y = new_labels > np.median(new_labels)
  noisy_y = (final_y ^ (np.random.uniform(size=final_y.shape) > 0.95))
  return noisy_y

def get_adversarial_labels(x_train_pqki, x_test_pqki, x_train, x_test, qubits, N_TRAIN):
  S_pqk, V_pqk = get_spectrum(
    tf.reshape(tf.concat([x_train_pqki, x_test_pqki], 0), [-1, len(qubits) * 3]))
  S_original, V_original = get_spectrum(
    tf.cast(tf.concat([x_train, x_test], 0), tf.float32), gamma=0.005)
  y_relabel = get_stilted_dataset(S_pqk, V_pqk, S_original, V_original)
  y_train_new, y_test_new = y_relabel[:N_TRAIN], y_relabel[N_TRAIN:]

  return y_train_new, y_test_new

def create_pqk_model(qubits):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='sigmoid', input_shape=[len(qubits) * 3,]))
    model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))
    return model

def create_and_train_pqk_model(qubits, x_train, y_train, x_test, y_test, N_TRAIN, N_TEST):
    pqk_model = create_pqk_model(qubits=qubits)
    pqk_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
              metrics=['accuracy'])
    print(f"Training with {N_TRAIN} samples")
    pqk_history = pqk_model.fit(tf.reshape(x_train, [N_TRAIN, -1]),
          y_train,
          batch_size=32,
          epochs=1000,
          verbose=0,
          validation_data=(tf.reshape(x_test, [N_TEST, -1]), y_test))

    return pqk_history

def create_fair_classical_model(dataset_dimension):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='sigmoid', input_shape=[dataset_dimension,]))
    model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))
    return model

def create_and_train_fair_classical_model(x_train, y_train, x_test, y_test, dataset_dimension):
    classical_model = create_fair_classical_model(dataset_dimension=dataset_dimension)
    classical_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
              metrics=['accuracy'])
    
    classical_history = classical_model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=1000,
          verbose=0,
          validation_data=(x_test, y_test))

    return classical_history

#### Methods for saving and loading data

def load_or_create_random_rots(n_qubits):
    __location__ = get_file_location()
    path = os.path.join(__location__, f'circuits/random_rots_nq{n_qubits}.npy')
    try:
        random_rots = np.load(path)
    except:
        random_rots = np.random.uniform(-2, 2, size=(n_qubits, 3))
        np.save(path, random_rots)
    return random_rots

def load_or_create_ideal_features(qubits, x_train, x_test, random_rots):
    __location__ = get_file_location()
    path_train = os.path.join(__location__, f'expectations/exp_train_N{len(x_train)}_qn{len(qubits)}_ideal.npy')
    path_test = os.path.join(__location__, f'expectations/exp_test_N{len(x_test)}_qn{len(qubits)}_ideal.npy')
    try:
        x_train_pqki = np.load(path_train)
        x_test_pqki = np.load(path_test)
    except:
        qi_x_train_circuits = prepare_pqk_circuits(qubits, x_train, random_rots)
        qi_x_test_circuits = prepare_pqk_circuits(qubits, x_test, random_rots)
        x_train_pqki = get_pqk_features(qubits, qi_x_train_circuits)
        x_test_pqki = get_pqk_features(qubits, qi_x_test_circuits)
        np.save(path_train, x_train_pqki)
        np.save(path_test, x_test_pqki)
    return x_train_pqki, x_test_pqki

def load_or_create_adversarial_labels(x_train_pqki, x_test_pqki, x_train, x_test, qubits, N_TRAIN):
    __location__ = get_file_location()
    path_train = os.path.join(__location__, f'data/y_train_N{N_TRAIN}_qn{len(qubits)}_adv.npy')
    path_test = os.path.join(__location__, f'data/y_test_N{len(x_test)}_qn{len(qubits)}_adv.npy')
    try:
        y_train_new = np.load(path_train)
        y_test_new = np.load(path_test)
    except:
        y_train_new, y_test_new = get_adversarial_labels(x_train_pqki, x_test_pqki, x_train, x_test, qubits, N_TRAIN)
        np.save(path_train, y_train_new)
        np.save(path_test, y_test_new)
    return y_train_new, y_test_new

def simulate_and_save_expectations(qubits, x_train, x_test, random_rots, p):
    print(f"prepare circuits with error rate {p}")
    qe_x_train_circuits = prepare_pqk_circuits(qubits, x_train, random_rots, p=p)
    qe_x_test_circuits = prepare_pqk_circuits(qubits, x_test, random_rots, p=p)
    print(f"get features with error rate {p}")
    x_train_pqkep = get_pqk_features(qubits, qe_x_train_circuits, noisy=True)
    x_test_pqkep = get_pqk_features(qubits, qe_x_test_circuits, noisy=True)
    __location__ = get_file_location()
    p_str = "{:.5f}".format(p)[2:]
    path_train = os.path.join(__location__, f'expectations/exp_train_N{len(x_train)}_qn{len(qubits)}_p{p_str}.npy')
    path_test = os.path.join(__location__, f'expectations/exp_test_N{len(x_test)}_qn{len(qubits)}_p{p_str}.npy')
    np.save(path_train, x_train_pqkep)
    np.save(path_test, x_test_pqkep)
    return x_train_pqkep, x_test_pqkep

def load_expectations(n_qubits, p, n_train, n_test):
    __location__ = get_file_location()
    p_str = "{:.5f}".format(p)[2:]
    path_train = os.path.join(__location__, f'expectations/exp_train_N{n_train}_qn{n_qubits}_p{p_str}.npy')
    path_test = os.path.join(__location__, f'expectations/exp_test_N{n_test}_qn{n_qubits}_p{p_str}.npy')
    x_train_pqkep = np.load(path_train)
    x_test_pqkep = np.load(path_test)
    return x_train_pqkep, x_test_pqkep

def load_or_simulate_expectations(qubits, x_train, x_test, random_rots, p):
    try:
      x_train_pqkep, x_test_pqkep = load_expectations(len(qubits), p, len(x_train), len(x_test))
    except:
      x_train_pqkep, x_test_pqkep = simulate_and_save_expectations(qubits, x_train, x_test, random_rots, p)
    return x_train_pqkep, x_test_pqkep

def save_model_history(n_train, n_test, n_qubits, p, history, i):
    p_str = "{:.5f}".format(p)[2:]
    __location__ = get_file_location()
    path = os.path.join(__location__, f'model_histories/N_TRAIN_{n_train}_N_TEST_{n_test}_qn{n_qubits}_p{p_str}_m{i}_val_accuracy.npy')
    np.save(path, history.history['val_accuracy'])

def load_model_history(n_train, n_test, n_qubits, p, i):
    p_str = "{:.5f}".format(p)[2:]
    __location__ = get_file_location()
    path = os.path.join(__location__, f'model_histories/N_TRAIN_{n_train}_N_TEST_{n_test}_qn{n_qubits}_p{p_str}_m{i}_val_accuracy.npy')
    try:
        history = np.load(path, allow_pickle=True)
    except:
        # Throw an error if the file is not found.
        raise FileNotFoundError(f'File {path} not found.')
    return history

def load_or_train_model_history(qubits, p, i, x_train, x_test, y_train, y_test, random_rots):
    try:
      history = load_model_history(len(x_train), len(x_test), len(qubits), p, i)
    except:
      x_train_pqkep, x_test_pqkep = load_or_simulate_expectations(qubits, x_train, x_test, random_rots, p)
      history = create_and_train_pqk_model(qubits, x_train_pqkep, y_train, x_test_pqkep, y_test, len(y_train), len(y_test))
      save_model_history(len(y_train), len(y_test), len(qubits), p, history, i)
      history = np.array(history.history['val_accuracy'])
    return history

def plot_model_histories(classical_history, ideal_history, errored_histories, errors, N_TRAIN, n_qubits, lwidth=0.9):
  __location__ = get_file_location()
  path = os.path.join(__location__, 'plots/')
  palette = colorpalette()
  colors = [palette.get('orange'), palette.get('charcoal'), palette.get('mantis'), palette.get('sandy'), palette.get('slate'), palette.get('cambridge'), palette.get('carmine'), palette.get('celadon')]
  plt.figure(figsize=(10, 5))

  plt.plot(classical_history, label='Classical', color = palette.get('celadon'), linewidth=lwidth)
  plt.plot(ideal_history, label='Ideal Quantum', color = palette.get('carmine'), linewidth=lwidth)
  for i, history in enumerate(errored_histories):
    plt.plot(history, label=f'Quantum, p={errors[i]}', color = colors[i], linewidth=lwidth)
  plt.legend()
  plt.xlabel('Epochs')
  plt.ylabel('Validation Accuracy')
  plt.title(f'Validation accuracies with {N_TRAIN} training samples and {n_qubits} qubits')
  err_min = min(errors)
  err_mins = "{:.5f}".format(err_min)[2:]
  err_max = max(errors)
  err_maxs = "{:.5f}".format(err_max)[2:]
  plt.savefig(path + f'acc-N{N_TRAIN}-qn{n_qubits}-e-{err_mins}-to-{err_maxs}.png', dpi = 500)
  