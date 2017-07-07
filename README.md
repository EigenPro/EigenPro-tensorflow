# EigenPro

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

## Experimental results

### Classification Error (MNIST)
In these experiments, EigenPro (Primal) achieves classification error 1.20%, after only 10 epochs. In comparison, Pegasos reaches error 1.22% after 80 epochs. Although the number of random features used by EigenPro (Random) and RF/DSGD is 6 * 10^4, same as the number of training points, methods using random features deliver generally worse performance. Specifically, RF/DSGD has error rate 1.80% after 20 epochs and Pegasos reaches error rate 1.63% after the same number of epochs.

<table>
  <tr>
    <th rowspan="2">#Epochs</th>
    <th colspan="4">Primal</th>
    <th colspan="4">Random Fourier Feature</th>
  </tr>
  <tr>
    <td colspan="2">EigenPro</td>
    <td colspan="2">Pegasos</td>
    <td colspan="2">EigenPro</td>
    <td colspan="2">RF/DSGD</td>
  </tr>
  <tr>
    <td></td>
    <td>train</td>
    <td>test</td>
    <td>train</td>
    <td>test</td>
    <td>train</td>
    <td>test</td>
    <td>train</td>
    <td>test</td>
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
    <td>EigenPro</td>
    <td>Pegasos</td>
    <td>EigenPro</td>
    <td>RF/DSGD</td>
  </tr>
  <tr>
    <td>One GTX Titan X (Maxwell)</td>
    <td>4.6s</td>
    <td>5.0s</td>
    <td>2.0s</td>
    <td>2.4s</td>
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
    <td>One GTX Titan X</td>
    <td>16.2s</td>
  </tr>
</table>
