# A Tutorial on Mix-and-Match Perturbation 

***Note 1: Due to IP issues, we are not able to release the code for human motion prediction at the moment. As soon as IP issues resolve, we will update the repo with motion prediction code.***

***Note 2: This is an example of Mix-and-Match perturbation on MNIST dataset. This code contains all the building blocks of Mix-and-Match.***


## Task: Conditional image completion.
In this experiment, the goal is to complete MNIST digits given partial observations. Note that the conditioning signal is strong enough such that a deterministic model can generate a digit image given the condition.


## Citation
If you find this work useful in your own research, please consider citing:

```
@inproceedings{mix-and-match-perturbation,
author={Aliakbarian, Sadegh and Saleh, Fatemeh Sadat and Salzmann, Mathieu and Petersson, Lars and Gould, Stephen},
title = {A Stochastic Conditioning Scheme for Diverse Human Motion Prediction},
booktitle = {Proceedings of the IEEE international conference on computer vision},
year = {2020}
}
```

## Running the code
To train the model, run:
```
python3 train.py
```
To sample given the trained model, run:
```
python3 sample.py
```

## Tutorial: Explaining how Mix-and-Match works!

![Sampling and Resampling](samples/sampling_resampling.png)

### Mix-and-Match VAE class
In this section, we explain different bits and pieces of [model.py](model.py). 

We first define the encoders and the decoder we used in this model. We have two encoders, one for the data that we one to learn the distribution of and one for the conditioning signal. Here is how we define the [data encoder](https://github.com/mix-and-match/mix-and-match-tutorial/blob/master/model.py#L17):
```
self.data_encoder = nn.Sequential(
            nn.Linear(args.input_dim, args.input_dim // 2),
            nn.BatchNorm1d(args.input_dim // 2),
            nn.ReLU(),
            nn.Linear(args.input_dim // 2, args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
        )  
```
This is a pretty simple network (based on fully connected layers) that maps a data of `args.input_dim` dimension to `args.hidden_dim`.  Note that the [encoder for the condition](https://github.com/mix-and-match/mix-and-match-tutorial/blob/master/model.py#L27) is completely identical to this network.

Similar to any VAE/CVAE, we have two layers that computes the paramters of the approximate posterior distribution:
```
self.mean = nn.Linear(args.hidden_dim, args.latent_dim)
self.std = nn.Linear(args.hidden_dim, args.latent_dim)
```

We now define the **Sampling** operation. In the forward pass, the function gets the input data and the conditioning signal. In this [function](https://github.com/mix-and-match/mix-and-match-tutorial/blob/master/model.py#L108), we first perform the Sampling operation:
```
self.sampled_indices = list(random.sample(range(0, self.args.hidden_dim), alpha))
self.complementary_indices = [i for i in range(self.args.hidden_dim) if i not in self.sampled_indices]
```
As shown in the figure in the beginning of this notebook, the sampling operation gets a vector length hidden_dim and a sampling rate alpha, and randomly samples alpha x hidden_dim (in code, alpha is the number of indices itself, not a rate) indices. It also creates a list of complementary indices that has not been sampled. These indices are later used to condition the encoder and the decoder of the VAE.

Given the input data and the sampled indices, we [encode](https://github.com/mix-and-match/mix-and-match-tutorial/blob/master/model.py#L63) the data:
```
h = self.data_encoder(data)
sampled_data = h[:, self.sampled_indices]
complementary_data = h[:, self.complementary_indices]        
```
Note, we do exactly the same for computing a representation out of the [conditioning signal](https://github.com/mix-and-match/mix-and-match-tutorial/blob/master/model.py#L52).
