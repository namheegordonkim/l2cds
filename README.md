# Learning to Correspond Dynamical Systems

This is a supplementary codebase for Learning to Correspond Dynamical Systems (L2CDS). More details can be found on the [project webpage](https://sites.google.com/view/l2cds).

## Requirements

This software package has been tested on the following environment:

* Ubuntu 18.04 LTS
* DART v6.6.2
* Python 3.6 or above

Following Python packages are required (not comprehensive):
* PyTorch
* PyDART
* matplotlib
* tqdm
* scikit-learn

### Docker / Singularity

A Dockerfile is provided for your convenience. The pre-built image is found on Docker Hub: https://hub.docker.com/r/namheegordonkim/l2cds

If you use Docker, use

```
docker pull namheegordonkim/l2cds
```

If you use Singularity, use

```
singularity pull docker://namheegordonkim/l2cds
```
