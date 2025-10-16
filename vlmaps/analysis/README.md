# VLMap distribution analysis tool

This is the main analysis tool for distributions resulting from 2d, 3d VLMaps and images.

## Input Data

### Input from 2D map

The script `input_formatter/input_from_2d_map.py` prepares the input for the analysis tool from the following two files:
* `--embeddings_file`: Map with embeddings, size: height x width x embedding_size
* `--labels_file`: Map with ground truth categories, size height x width
* `--output_path`: The path where to save  output .npy file in format that can be passed to the analysis tool.

**Example**

Download example data from [Google Drive](https://drive.google.com/drive/folders/1itXf0coCGCPzV5bK3zlB9fzqlHCyh2aY?usp=sharing). 
(Also contains the output file from execution of this script.)

Example running script: `prepare_input_2d.sh`.


## Usage

TBA

## Fitter disributions

All
['_fit', 'alpha', 'anglit', 'arcsine', 'argus', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy', 'chi', 'chi2', 'cosine', 'crystalball', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponpow', 'exponweib', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'gamma', 'gausshyper', 'genexpon', 'genextreme', 'gengamma', 'genhalflogistic', 'genhyperbolic', 'geninvgauss', 'genlogistic', 'gennorm', 'genpareto', 'gibrat', 'gilbrat', 'gompertz', 'gumbel_l', 'gumbel_r', 'halfcauchy', 'halfgennorm', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'kappa3', 'kappa4', 'ksone', 'kstwo', 'kstwobign', 'laplace', 'laplace_asymmetric', 'levy', 'levy_l', 'levy_stable', 'loggamma', 'logistic', 'loglaplace', 'lognorm', 'loguniform', 'lomax', 'maxwell', 'mielke', 'moyal', 'nakagami', 'ncf', 'nct', 'ncx2', 'norm', 'norminvgauss', 'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rayleigh', 'rdist', 'recipinvgauss', 'reciprocal', 'rice', 'rv_continuous', 'rv_histogram', 'semicircular', 'skewcauchy', 'skewnorm', 'studentized_range', 't', 'trapezoid', 'trapz', 'triang', 'truncexpon', 'truncnorm', 'truncweibull_min', 'tukeylambda', 'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_max', 'weibull_min', 'wrapcauchy']

Common
['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm', 'norm', 'powerlaw', 'rayleigh', 'uniform']