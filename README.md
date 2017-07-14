# EigenPro

## Intro
EigenPro is a preconditioned (stochastic) gradient descent iteration that accelerates the convergence on minimizing (typically) kernel least squares, defined as

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=L(\alpha)&space;=&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;{&space;(\left&space;\langle&space;\alpha,&space;x_i&space;\right&space;\rangle_\mathcal{H}&space;-&space;y_i)^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(\alpha)&space;=&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;{&space;(\left&space;\langle&space;\alpha,&space;x_i&space;\right&space;\rangle_\mathcal{H}&space;-&space;y_i)^2}" title="L(\alpha) \doteq \frac{1}{n} \sum_{i=1}^{n} { (\left \langle \alpha, x_i \right \rangle_\mathcal{H} - y_i)^2}" /></a>
</p>

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha^*&space;=&space;\arg\min_{\alpha&space;\in&space;\mathcal{H}}&space;{L(\alpha)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha^*&space;=&space;\arg\min_{\alpha&space;\in&space;\mathcal{H}}&space;{L(\alpha)}" title="\alpha^* = \arg\min_{\alpha \in \mathcal{H}} {L(\alpha)}" /></a>
</p>

In a simple linear setting (e.g. with linear kernel), the target solution is a vector such that

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;\langle&space;\alpha,&space;x_i&space;\right&space;\rangle_\mathcal{H}&space;\doteq&space;\alpha^T&space;x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;\langle&space;\alpha,&space;x_i&space;\right&space;\rangle_\mathcal{H}&space;\doteq&space;\alpha^T&space;x_i" title="\left \langle \alpha, x_i \right \rangle_\mathcal{H} \doteq \alpha^T x_i" /></a>
</p>

The corresponding standard gradient descent iteration is hence,

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;\leftarrow&space;\alpha&space;-&space;\eta&space;(H&space;\alpha&space;-&space;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;\leftarrow&space;\alpha&space;-&space;\eta&space;(H&space;\alpha&space;-&space;b)" title="\alpha \leftarrow \alpha - \eta (H \alpha - b)" /></a>
</p>

where H is the covariance matrix and b is the label vector transformed by the design matrix X.

Note that when adopting a smooth kernel (e.g. linear kernel, Gaussian kernel), the covariance matrix H has exponential eigendecay.
This normally makes convergence along small eigendirections of H overly slow.
To address the slow convergence, we construct EigenPro preconditioner using the approximate top eigensystem of H,
which can be efficiently calculated due to the fast eigendecay.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;\doteq&space;I&space;-&space;\sum_{i=1}^k&space;{(1&space;-&space;\frac{\lambda_{k&plus;1}(H)}&space;{\lambda_i(H)})&space;e_i(H)&space;e_i(H)^T}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P&space;\doteq&space;I&space;-&space;\sum_{i=1}^k&space;{(1&space;-&space;\frac{\lambda_{k&plus;1}(H)}&space;{\lambda_i(H)})&space;e_i(H)&space;e_i(H)^T}" title="P \doteq I - \sum_{i=1}^k {(1 - \frac{\lambda_{k+1}(H)} {\lambda_i(H)}) e_i(H) e_i(H)^T}" /></a>
</p>

The EigenPro iteration then runs as follows,
<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\alpha&space;\leftarrow&space;\alpha&space;-&space;(\eta&space;\frac{\lambda_1(H)}{\lambda_{k&plus;1}(H)})&space;P(H\alpha&space;-&space;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha&space;\leftarrow&space;\alpha&space;-&space;(\eta&space;\frac{\lambda_1(H)}{\lambda_{k&plus;1}(H)})&space;P(H\alpha&space;-&space;b)" title="\alpha \leftarrow \alpha - (\eta \frac{\lambda_1(H)}{\lambda_{k+1}(H)}) P(H\alpha - b)" /></a>
</p>

Due to the exponential eigendecay, EigenPro substantially increases the step size and greatly accelerates the convergence.
When \alpha is in a kernel space of infinite dimension, EigenPro iteration still achieves considerable acceleration over standard gradient methods like Pegasos.

 

## Install Keras (>=2.0.2) and Tensorflow (>=1.2.1)
```
pip install tensorflow tensorflow-gpu keras
```
Follow the [Tensorflow installation guide](https://www.tensorflow.org/install/install_linux) for Virtualenv setup.

## Run experiments
The experiments will compare Pegasos, Kernel EigenPro, Random Fourier Feature with linear SGD, and Random Fourier Feature with EigenPro on MNIST.
```
python run_expr.py
```

Besides, users can pass the flag "--kernel" to choose different kernels such like Gaussian, Laplace, and Cauchy.
```
python run_expr.py --kernel=Laplace
```
Note that we have only implemented the random Fourier feature for Gaussian kernel.

## Use the EigenPro iteration
The EigenPro iteration can be called through a Keras Model. Please read this short [Keras tutorial](https://keras.io/getting-started/sequential-model-guide/)
to get familiar with its components.

The EigenPro iteration is integrated in the two optimizers, SGD and PSGD. The former works with finite dimension feature map like random Fourier feature; the latter works in an RKHS (reproducing kernel Hilbert space) related to a kernel function. Note the latter requires appending a sample id (used during training) to each data sample,
```
x_train = utils.add_index(x_train)
```

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
In these experiments, EigenPro (Primal) achieves classification error 1.20%, after only 10 epochs. In comparison, Pegasos reaches error 1.22% after 80 epochs. Although the number of random features used by EigenPro (Random) and RF/DSGD is 6 * 10^4, same as the number of training points, methods using random features deliver generally worse performance. Specifically, RF/DSGD has error rate 1.80% after 20 epochs and Pegasos reaches error rate 1.63% after the same number of epochs.

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
    <td>0.06%</td>
    <td>1.42%</td>
    <td>4.01%</td>
    <td>4.35%</td>
    <td>0.11%</td>
    <td>1.61%</td>
    <td>4.03%</td>
    <td>4.31%</td>
  </tr>
  <tr>
    <td>5</td>
    <td>0.01%</td>
    <td>1.25%</td>
    <td>1.58%</td>
    <td>2.32%</td>
    <td>0.02%</td>
    <td><b>1.51%</b></td>
    <td>1.64%</td>
    <td>2.41%</td>
  </tr>
  <tr>
    <td>10</td>
    <td>0.0%</td>
    <td><b>1.20%</b></td>
    <td>0.89%</td>
    <td>1.91%</td>
    <td>0.0%</td>
    <td>1.55%</td>
    <td>2.02%</td>
    <td>1.97%</td>
  </tr>
  <tr>
    <td>20</td>
    <td>0.0%</td>
    <td>1.23%</td>
    <td>0.40%</td>
    <td>1.63%</td>
    <td>0.0%</td>
    <td>1.54%</td>
    <td>0.48%</td>
    <td>1.80%</td>
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
    <td align="center">16.2s</td>
  </tr>
</table>
