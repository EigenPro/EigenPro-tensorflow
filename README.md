# EigenPro

## Intro
EigenPro is a preconditioned (stochastic) gradient descent iteration proposed in [the paper](https://arxiv.org/abs/1703.10622):
```
Ma, Siyuan, and Mikhail Belkin. Diving into the shallows:
a computational perspective on large-scale shallow learning.
In NIPS, 2017.
```

It accelerates the convergence of SGD iteration when minimizing linear and kernel least squares, defined as

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\arg&space;\min_{{\pmb&space;\alpha}&space;\in&space;\mathcal{H}}&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;{&space;(\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;-&space;y_i)^2}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\arg&space;\min_{{\pmb&space;\alpha}&space;\in&space;\mathcal{H}}&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;{&space;(\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;-&space;y_i)^2}" title="\arg \min_{{\pmb \alpha} \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^{n} { (\left \langle {\pmb \alpha}, {\pmb x}_i \right \rangle_\mathcal{H} - y_i)^2}" /></a>
</p>

where
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{({\pmb&space;x}_i,&space;y_i)\}_{i=1}^n" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\{({\pmb&space;x}_i,&space;y_i)\}_{i=1}^n" title="\{({\pmb x}_i, y_i)\}_{i=1}^n" /></a>
is the labeled training data. Let
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X&space;\doteq&space;({\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n)^T,&space;{\pmb&space;y}&space;\doteq&space;(y_1,&space;\ldots,&space;y_n)^T" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;X&space;\doteq&space;({\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n)^T,&space;{\pmb&space;y}&space;\doteq&space;(y_1,&space;\ldots,&space;y_n)^T" title="X \doteq ({\pmb x}_1, \ldots, {\pmb x}_n)^T, {\pmb y} \doteq (y_1, \ldots, y_n)^T" /></a>
.


Consdier the linear setting where 
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{H}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\mathcal{H}" title="\mathcal{H}" /></a>
is a vector space and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;\doteq&space;{\pmb&space;\alpha}^T&space;{\pmb&space;x}_i" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;\doteq&space;{\pmb&space;\alpha}^T&space;{\pmb&space;x}_i" title="\left \langle {\pmb \alpha}, {\pmb x}_i \right \rangle_\mathcal{H} \doteq {\pmb \alpha}^T {\pmb x}_i" /></a>
. The corresponding standard gradient descent iteration is hence,

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex={\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;\eta&space;(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" target="_blank"><img src="https://latex.codecogs.com/png.latex?{\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;\eta&space;(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" title="{\pmb \alpha} \leftarrow {\pmb \alpha} - \eta (H {\pmb \alpha} - {\pmb b})" /></a>
</p>

where
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;H&space;\doteq&space;X^T&space;X" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;H&space;\doteq&space;X^T&space;X" title="H \doteq X^T X" /></a>
is the covariance matrix and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\pmb&space;b}&space;\doteq&space;X^T{\pmb&space;y}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;{\pmb&space;b}&space;\doteq&space;X^T{\pmb&space;y}" title="{\pmb b} \doteq X^T{\pmb y}" /></a>
. The step size is automatically set as
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\eta&space;\leftarrow&space;1.5&space;\cdot&space;\lambda_1(H)^{-1}" target="_blank"><img align="center" src="https://latex.codecogs.com/gif.latex?\inline&space;\eta&space;\leftarrow&space;1.5&space;\cdot&space;\lambda_1(H)^{-1}" title="\eta \leftarrow 1.5 \cdot \lambda_1(H)^{-1}" /></a>
to ensure fast convergence. Note that the top eigenvalue of the covariance is calculated approximately.
We then construct EigenPro preconditioner P using the approximate top eigensystem of H,
which can be efficiently calculated when H has fast eigendecay.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;\doteq&space;I&space;-&space;\sum_{i=1}^k&space;{(1&space;-&space;\tau&space;\frac{\lambda_{k&plus;1}(H)}&space;{\lambda_i(H)})&space;{\pmb&space;e}_i(H)&space;{\pmb&space;e}_i(H)^T}" target="_blank"><img src="https://latex.codecogs.com/png.latex?P&space;\doteq&space;I&space;-&space;\sum_{i=1}^k&space;{(1&space;-&space;\tau&space;\frac{\lambda_{k&plus;1}(H)}&space;{\lambda_i(H)})&space;{\pmb&space;e}_i(H)&space;{\pmb&space;e}_i(H)^T}" title="P \doteq I - \sum_{i=1}^k {(1 - \tau \frac{\lambda_{k+1}(H)} {\lambda_i(H)}) {\pmb e}_i(H) {\pmb e}_i(H)^T}" /></a>
</p>

Here we select
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tau&space;\leq&space;1" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\tau&space;\leq&space;1" title="\tau \leq 1" /></a>
to counter the negative impact of eigensystem approximation error on convergence.
The EigenPro iteration then runs as follows,

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex={\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;(\eta&space;\frac{\lambda_1(H)}{\lambda_{k&plus;1}(H)})&space;P(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" target="_blank"><img src="https://latex.codecogs.com/png.latex?{\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;(\eta&space;\frac{\lambda_1(H)}{\lambda_{k&plus;1}(H)})&space;P(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" title="{\pmb \alpha} \leftarrow {\pmb \alpha} - (\eta \frac{\lambda_1(H)}{\lambda_{k+1}(H)}) P(H {\pmb \alpha} - {\pmb b})" /></a>
</p>

With larger
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\lambda_1(H)&space;/&space;\lambda_{k&plus;1}(H)" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\lambda_1(H)&space;/&space;\lambda_{k&plus;1}(H)" title="\lambda_1(H) / \lambda_{k+1}(H)" /></a>
, EigenPro iteration yields higher convergence acceleration over standard (stochastic) gradient descent.
This is especially critical in the kernel setting where (widely used) smooth kernels have exponential eigendecay.
Note that in such setting 
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{H}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\mathcal{H}" title="\mathcal{H}" /></a> 
is typically an RKHS (reproducing kernel Hilbert space) of infinite dimension. Thus it is necessary to parametrize the (approximate) solution in a subspace of finite dimension (e.g. 
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathrm{span}_{{\pmb&space;x}&space;\in&space;\{&space;{\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n&space;\}}&space;\{&space;k(\cdot,&space;{\pmb&space;x})&space;\}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\mathrm{span}_{{\pmb&space;x}&space;\in&space;\{&space;{\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n&space;\}}&space;\{&space;k(\cdot,&space;{\pmb&space;x})&space;\}" title="\mathrm{span}_{{\pmb x} \in \{ {\pmb x}_1, \ldots, {\pmb x}_n \}} \{ k(\cdot, {\pmb x}) \}" /></a>
).
See [the paper](https://arxiv.org/abs/1703.10622) for more details on the kernel setting and some theoretical results.



## Requirements: Tensorflow (>=1.2.1) and Keras (=2.0.8)
```
pip install tensorflow tensorflow-gpu keras
```
Follow the [Tensorflow installation guide](https://www.tensorflow.org/install/install_linux) for Virtualenv setup.


## Running experiments
The experiments will compare Pegasos, Kernel EigenPro, Random Fourier Feature with linear SGD, and Random Fourier Feature with EigenPro on MNIST.
```
python run_expr.py
```

Besides, users can pass the flag "--kernel" to choose different kernels such like Gaussian, Laplace, and Cauchy.
```
python run_expr.py --kernel=Laplace
```
Note that we have only implemented the random Fourier feature for the Gaussian kernel.


## An example of building and training a kernel model
First, let's import the related Keras and kernel components,
```
from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K

from layers import KernelEmbedding
from optimizers import PSGD
```

Please read this short [Keras tutorial](https://keras.io/getting-started/sequential-model-guide/)
to get familiar with its components.
Then we can create the input layer,
```
import mnist
import utils
(x_train, y_train), (x_test, y_test) = mnist.load()
n, D = x_train.shape
ix = Input(shape=(D+1,), dtype='float32', name='indexed-feat')
x, index = utils.separate_index(ix) # features, sample_id
```

Note that the initialization of PSGD (SGD optimizer for primal kernel method) needs a tensor that records the sample ID.
Therefore we preprocess each sample by appending its sample id after its feature vector (ix).
The KernelEmbedding layer is a non-trainable layer that maps input feature vector (x) to kernel features (kfeat) 
with a given kernel function (in kernels.py) 
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\pmb&space;x}&space;\rightarrow&space;(k({\pmb&space;x},&space;{\pmb&space;x}_1),&space;\ldots,&space;k({\pmb&space;x},&space;{\pmb&space;x}_n))^T" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;{\pmb&space;x}&space;\rightarrow&space;(k({\pmb&space;x},&space;{\pmb&space;x}_1),&space;\ldots,&space;k({\pmb&space;x},&space;{\pmb&space;x}_n))^T" title="{\pmb x} \rightarrow (k({\pmb x}, {\pmb x}_1), \ldots, k({\pmb x}, {\pmb x}_n))^T" /></a>
,
```
kfeat = KernelEmbedding(kernel, x_train,
                        input_shape=(D,))(x)
```

Since the kernel least squares is essentially a linear least squares model using the kernel features,
we create a trainable Dense (linear) layer for the kernel features to predict corresponding labels.
```
y = Dense(num_classes, input_shape=(n,),
          activation='linear',
          kernel_initializer='zeros',
          use_bias=False)(kfeat)
```

Thus the Keras model can be created using the input tensor (ix) and the prediction tensor (y).
Also, calling the compile(...) method to specify the loss function and optimizer for training,
as well as the metrics for evaluation.
```
model = Model(ix, y)
model.compile(loss='mse',
              optimizer=PSGD(pred_t=y, index_t=index, eta=5.),
              metrics=['accuracy'])
```

The training can be performed by calling the method fit(...),
```
model.fit(utils.add_index(x_train), y_train,
          batch_size=256, epochs=10, verbose=0,
          validation_data=(utils.add_index(x_test), y_test))
```

It will run for 10 epochs using mini-batches of size 256. Note that utils.add\_index(...) will append
the sample id to each sample feature vector.

To evaluate the training result, we can call the method evaluate(...),
```
scores = model.evaluate(utils.add_index(x_test), y_test, verbose=0)
```
where scores[0] is the L2 loss (mse) and scores[1] the accuracy on the testing set.



## Using the EigenPro iteration
The EigenPro iteration can be called through a Keras Model.
It is integrated in the two optimizers, SGD and PSGD. The former works with a finite dimension feature map like random Fourier feature; the latter works in an RKHS related to a kernel function. Note the latter requires appending a sample id (used during training) to each data sample.

By default, the optimizers use standard (stochastic) gradient descents. To enable EigenPro iteration, pass parameter eigenpro\_f to the optimizer, such like
```
PSGD(... , eta=scale*eta, eigenpro_f=f)
```
where scale is the eigenvalue ratio used to increase the step size and f is the EigenPro preconditioner (or more specifically, I - P. See the intro section).
Both can be calculated using utils.py,
```
f, scale = utils.asm_eigenpro_f(... , in_rkhs=True)
```
Here flag in\_rkhs indicates if the calculation is for PSGD (infinite dimension RKHS) or SGD (finite dimension vector space).
The function will use truncated randomized SVD (for small dataset) or Nystrom based SVD (for large dataset) to calcualte the approximate top eigensystem of the covariance.

Note that the optimizer should be connected to a Keras model,
```
model.compile(loss='mse', optimizer=PSGD(...), metrics=['accuracy'])
```

After the optimizer is appropriately initialized for a model, the EigenPro iteration will be used through model training,
```
model.fit(x_train, y_train)
```


## Reference experimental results

### Classification Error (MNIST)
In these experiments, EigenPro (Primal) achieves classification error 1.22% using only 10 epochs. For comparison, Pegasos reaches the same error after 80 epochs. Although the number of random features used by EigenPro (Random) and RF/DSGD is 6 * 10^4, same as the number of training points, methods using random features deliver generally worse performance. Specifically, RF/DSGD has error rate 1.75% after 20 epochs and Pegasos reaches error rate 1.63% after the same number of epochs.

<table>
  <tr>
    <th rowspan="2">#Epochs</th>
    <th colspan="4">Primal</th>
    <th colspan="4">Random Fourier Feature</th>
  </tr>
  <tr>
    <td colspan="2" align="center">EigenPro</td>
    <td colspan="2" align="center">Pegasos</td>
    <td colspan="2" align="center">EigenPro</td>
    <td colspan="2" align="center">RF/DSGD</td>
  </tr>
  <tr>
    <td></td>
    <td align="center">train</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">test</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0.43%</td>
    <td>1.75%</td>
    <td>4.01%</td>
    <td>4.35%</td>
    <td>0.39%</td>
    <td>1.88%</td>
    <td>4.00%</td>
    <td>4.35%</td>
  </tr>
  <tr>
    <td>5</td>
    <td>0.02%</td>
    <td>1.26%</td>
    <td>1.58%</td>
    <td>2.32%</td>
    <td>0.05%</td>
    <td><b>1.48%</b></td>
    <td>1.70%</td>
    <td>2.51%</td>
  </tr>
  <tr>
    <td>10</td>
    <td>0.0%</td>
    <td><b>1.22%</b></td>
    <td>0.89%</td>
    <td>1.91%</td>
    <td>0.01%</td>
    <td>1.49%</td>
    <td>0.98%</td>
    <td>2.09%</td>
  </tr>
  <tr>
    <td>20</td>
    <td>0.0%</td>
    <td>1.23%</td>
    <td>0.40%</td>
    <td>1.63%</td>
    <td>0.0%</td>
    <td>1.48%</td>
    <td>0.48%</td>
    <td>1.75%</td>
  </tr>
</table>


### Training Time per Epoch

<table>
  <tr>
    <th rowspan="2">Computing<br>Resource</th>
    <th colspan="2">Primal</th>
    <th colspan="2">Random Fourier Feature</th>
  </tr>
  <tr>
    <td align="center">EigenPro</td>
    <td align="center">Pegasos</td>
    <td align="center">EigenPro</td>
    <td align="center">RF/DSGD</td>
  </tr>
  <tr>
    <td>One GTX Titan X (Maxwell)</td>
    <td align="center">5.0s</td>
    <td align="center">4.6s</td>
    <td align="center">2.4s</td>
    <td align="center">2.0s</td>
  </tr>
  <tr>
    <td>One GTX Titan Xp (Pascal)</td>
    <td align="center">3.0s</td>
    <td align="center">2.7s</td>
    <td align="center">1.6s</td>
    <td align="center">1.4s</td>
  </tr>
</table>

### EigenPro Preprocessing Time
In our experiments we construct the EigenPro preconditioner by computing the top 160 approximate eigenvectors for a subsample matrix with 4800 points using Randomized SVD (RSVD).

<table>
  <tr>
    <th>Computing<br>Resource</th>
    <th>RSVD Time<br>(k = 160, m = 4800)</th>
  </tr>
  <tr>
    <td>One GTX Titan X (Maxwell)</td>
    <td align="center">18.1s</td>
  </tr>
  <tr>
    <td>One GTX Titan Xp (Pascal)</td>
    <td align="center">17.4s</td>
  </tr>
</table>
