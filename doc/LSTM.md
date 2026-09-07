## Recurrent Neural Networks (RNNs)

Unlike traditional feed-forward networks (which process each input in isolation and immediately forget previous data), RNNs incorporate a short-term memory component in the hidden state.

### Limitations of RNNs

* **Vanishing Gradient**: During training via Backpropagation Through Time (BPTT), if the sequences are long, the error gradient tends to approach zero. The network stops learning from distant temporal dependencies, causing underfitting.
* **Exploding Gradient**: The gradient grows exponentially with each iteration, making the network unstable and prone to erratic behavior or overfitting.
* **Limited Memory**: Standard RNNs can only remember information from the immediate past.
* **Computational Slowness**: Strictly sequential processing (one step at a time) prevents parallelization and makes training on long texts slow and extremely computationally demanding.

## LSTM

**The Memory Cell**: A special channel in the hidden layer designed to store and propagate information over long periods.


**The Three Gate Mechanisms**:

* **Forget Gate**: Decides which past information to discard or remove from the memory cell.
* **Input Gate**: Decides which new incoming information should be stored and updated in the cell.
* **Output Gate**: Decides which information from the memory cell should be propagated as output and hidden state.

### Training Pipeline

#### 1\. Data Preparation and Preprocessing

**Sequential data** are types of data (such as words, sentences, or time series) in which individual elements are not independent but are correlated according to complex syntactic, semantic, or temporal rules.
The process of feeding and training sequential data into an LSTM network follows defined steps, and managing the data format is the most critical phase when working with temporal sequences:

* **Set Splitting**: Data are divided into training and validation sets following a temporal logic to avoid data leakage.
* **Windowing**: The continuous time series is broken into fixed-length sequences called “windows.” Each window contains the input features (X) (composed of the (n) previous time steps) and the corresponding target (y) (the next value to predict).
* **Normalization**: To stabilize computations and prevent numerical instability, data are normalized (for example via min-max scaling between 0 and 1, or zero-centered normalization by computing the mean and standard deviation for each feature).
* **Sorting and Padding/Truncation**: Since real sequences can have different lengths, the pipeline pads with dummy values up to the maximum length (option “longest”) or truncates based on the shortest sequence (option “shortest”). To optimize computational performance, it is good practice to sort sequences by length before creating mini-batches, reducing the amount of unnecessary padding.
* **Batching**: Data are organized into mini-batches (for example of size 32 or 64 samples) to promote efficient use of hardware memory and compute stable gradient updates.

#### 2\. Model Architecture Definition

The building blocks of the network are assembled by defining the dimensionality of the flows:

* **Sequence Input Layer**: Constitutes the entry point for the sequences and performs any automatic normalization of the input data.
* **LSTM / BiLSTM Layer**: Processes the inputs by passing them through the gate structures (forget, input, and output) to update the cell state ((c\_t)) and the hidden state ((h\_t)). Bidirectional LSTM (BiLSTM) layers can be used to allow the network to consider the sequence in both directions (past and future). In deeper architectures, multiple LSTM layers are stacked (setting sequence passing between the various layers) alternating with Dropout layers to counteract overfitting.
* **Output Layer**: Includes a fully connected (Dense) layer that maps the final state to the desired output. In regression tasks, an inverse normalization layer can be included to restore the real scale of the target directly during prediction.

#### 3\. Model Compilation (Loss and Optimizer)

The mathematical elements that guide learning are configured:

* **Loss Function**: Defined according to the specific task. Huber loss is used for regression tasks (because it effectively balances mean absolute error MAE and mean squared error MSE in the presence of outliers) or categorical cross-entropy for classification.
* **Optimizer**: Algorithms such as Adam or AdamW dynamically adjust the network weights based on a learning rate and regularization parameters.

#### 4\. Training and Backpropagation Through Time (BPTT)

The network runs training cycles (epochs) to update the parameters:

* **Backpropagation Through Time (BPTT)**: At each iteration, the network computes the committed error and propagates the gradients backward along the temporal steps of the sequence to update the neuron weights (shared across the different LSTM steps).
* **Monitoring Callbacks (Early Stopping)**: During the fit cycle, callbacks such as Early Stopping are used to monitor metrics on the validation set (e.g., the MAE value or the loss). If performance does not improve for a certain number of epochs, training is interrupted to avoid overfitting and save resources.

#### 5\. Evaluation and Prediction (Inference)

* **Evaluation**: Performance on the validation set is quantified by calculating metrics such as MAE, MSE, or classification accuracy.
* **Inference and State Management**: In field prediction, LSTMs can operate in “stateful” mode, i.e., maintaining the cell state and hidden state in memory between one prediction and the next. This capability allows dynamic and incremental step-by-step predictions on continuous or streaming data flows, without the need to recompute the entire history.

### Validation Metrics

The metrics used to evaluate the learning and performance of an LSTM model change depending on the type of task for which the network is trained, distinguishing primarily between **regression** tasks (such as time series forecasting) and **classification** tasks (such as sequence or video classification).

#### 1\. Metrics for Regression Tasks

In typical LSTM applications, such as predicting the next point in a time series, performance metrics evaluate the discrepancy between the continuous real values and the predicted ones:

* **MAE (Mean Absolute Error)**: Measures the average of the absolute differences between the real values and the model’s predictions. It is one of the most used metrics to quantify how precise the network is in predicting the next point of the sequence.
* **MSE (Mean Squared Error)**: Calculates the average of the squared errors. By squaring the differences, this metric penalizes large errors much more severely.
* **RMSE (Root Mean Squared Error)**: It is indicated as the standard reference metric for evaluating the accuracy and stability of sequential predictions over different time horizons.
* **Huber Loss**: Although it is a loss function used to guide weight optimization during training, it is monitored as a performance indicator. It is particularly effective because it acts as a balanced combination of MAE and MSE, showing great robustness in the presence of data with outliers or anomalous values.

#### 2\. Metrics for Classification Tasks

When the LSTM architecture is designed for classification tasks (from sequence to label), the network usually ends with a fully connected layer and a Softmax layer to produce class probabilities:

* **Classification Accuracy**: Represents the fundamental metric in this scenario and measures the proportion of temporal sequences or samples that have been correctly classified by the model relative to the overall total.

## Transformer

Transformers overcome the memory and computational limitations typical of RNNs by completely eliminating the need for sequential hidden states.
**(Self-Attention)**: Allows the entire data sequence to be processed in parallel. The model directly computes the interdependencies between all elements of the sequence, using positional encoding to remember the correct order.
**Advantages of Parallelism**:

* Definitive resolution of the vanishing gradient problem: since there is no sequential propagation through time, gradients flow freely to all network weights.
* Training of models of significantly larger size in fractions of the time required by an RNN.
* Ideal optimization for parallel computation on dedicated hardware (GPUs).

## When to Use LSTM

**LSTM** networks retain a crucial and irreplaceable role for processing continuous real-time data streams, especially on edge devices or with limited computational and energy resources.

* **Incremental State Update ((O(1)) vs (O(n)) per step)**: The recurrent update of an LSTM cell has computational and memory complexity equal to (O(1)) with respect to sequence length at each time step. In contrast, a Transformer attention block requires processing the relationships between the new token and all previous ones (complexity equal to (O(n)) per step, even with KV memory caching), leading to a linear growth in memory consumption and latency as the streaming flow continues indefinitely.
* **Energy Efficiency and Edge Computing**: A recent study (2025) focused on energy efficiency notes that the computational footprint and high memory consumption of Transformers make them complex to apply at scale for edge temporal classification tasks. Recurrent networks such as LSTMs require far fewer parameters, translating into significantly lower energy consumption, ideal for IoT and microelectronics.
* **Computational Limitations of Self-Attention**: An analysis published on **PMC/NIH** highlights that the computational complexity of the classic self-attention mechanism grows quadratically ((O(n^2))) with sequence length. In distributed architectures or resource-limited nodes (such as edge or fog computing), this represents a critical bottleneck that recurrent networks structurally avoid.
* **Adaptability to Lightweight Infrastructures**: A systematic review published on **Preprints.org** (2026) compares Transformers and LSTMs for time series forecasting, concluding that although Transformers excel on long high-resolution historical windows, LSTMs remain the optimal choice for lightweight and fast applications to be deployed on limited infrastructure hardware.

**In summary**, LSTMs remain highly relevant whenever you need efficient, incremental processing of continuous data streams, especially on resource-constrained or edge devices. Their constant-time state update and low memory footprint make them more practical than Transformers in real-time, low-power, or long-running streaming scenarios. While modern Generative AI (predominantly Transformer-based) excels at large-scale parallel training and capturing long-range dependencies, LSTMs often provide a better trade-off in terms of latency, energy consumption, and deployment simplicity. Choose LSTMs when efficiency and continuous inference matter most; prefer Transformers when maximum modeling capacity and parallelization are the priority.
