# Anomaly-Detection
Anomaly detection methods and implementation.

# Preprocessing
* [utils.py](util/utils.py) - Trajectory matrix/ rolling window representation of time series data. 

# Algorithms
* [LSTM Autoencoder](notebook/machine_temperature_system_failure.ipynb)
   * Long short-term memory (LSTM) is a recurrent neural network (RNN) architecture, used to process entire sequences of data.
   * An autoencoder learns a representation for a set of data in an unsupervised manner by training the network to ignore signal “noise”.
   * The reconstruction error serves as an indicator to the extent of a data instance being an anomaly.

# Datasets
* [Numenta Anomaly Benchmark](https://github.com/numenta/NAB)
   * machine_temperature_system_failure.csv
      * Temperature sensor data of an internal component of a large, industrial machine.
      * The first anomaly is a planned shutdown of the machine.
      * The second anomaly is difficult to detect and directly led to the third anomaly, a catastrophic failure of the machine.
