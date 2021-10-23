# Learning to Correspond Dynamical Systems

This is a supplementary codebase for Learning to Correspond Dynamical Systems (L2CDS). More details can be found on the [project webpage](https://sites.google.com/view/l2cds).

## Requirements

This software package has been tested on the following environment:

* Ubuntu 18.04 LTS
* DART v6.6.2
* Python 3.7 or above

Following Python packages are required (not comprehensive):
* scikit-learn==0.22.0 (importing pre-computed results will break otherwise)
* PyTorch (torch==1.19.0+cu111 recommended)
* PyDart2
* matplotlib
* tqdm
* stable-baselines3

### With Docker / Singularity

**NOTE:** Visualization with GLUT window isn't really supported inside containers. For visualization, I recommend using the non-container route, although it involves more headache.

A Dockerfile is provided for your convenience. The pre-built image is found on Docker Hub: https://hub.docker.com/r/namheegordonkim/l2cds

If you use Docker, use

```
docker pull namheegordonkim/l2cds
```

After pulling the latest `l2cds` image, please mount the local repo directory to `/l2cds` within the container, i.e.

```
docker run -it --mount type=bind,source=$(pwd),target=/l2cds namheegordonkim/l2cds
```

If you use Singularity, use

```
singularity pull docker://namheegordonkim/l2cds
```

### Without Docker / Singularity

1. Ensure that your OS-level Python is version 3.7 or above.

2. Follow the instructions [here](https://dartsim.github.io/install_dart_on_ubuntu.html#build-and-install-dart) to build and install DART v6.6.2. That is, ensure to run `git checkout tags/v6.6.2` before building with `cmake` and `make`.

3. Build [PyDart2](https://github.com/sehoonha/pydart2) from source, i.e. do something like:

```
git clone https://github.com/sehoonha/pydart2.git
cd pydart2
python setup.py build build_ext
python setup.py develop
```

**NOTE:** The way relative import works in Python 3.7+ seems to mess with the way SWIG generates bindings. In case any error regarding `_pydart2_api` shows up, do the following
* Locate `pydart2/pydart2.api.py`
* Change relative import statement `from . import _pydart2_api` to `import _pydart2_api`.

4. Install [dart-env](https://github.com/DartEnv/dart-env.git). You will want to build it from source in develop mode using `pip install -e ".[dart]"`, i.e.

```
git clone git clone https://github.com/DartEnv/dart-env.git
cd dart-env
pip install -e ".[dart]"
```

5. Install the rest of the requirements.

## Running Experiments

* When running within the docker container, use `python3`. If not, use `python3` or `python`, whichever works. 

* The training scripts will dump snapshots at a specified interval (`save_every` in config files), and these will be used for computing correspondence.

* By default, the config files specify experiments to run for large numbers of iterations. However, you can babysit for the loss value and terminate the experiment whenever, and use the latest dumped artifacts.

### Shapes Experiments

#### Make shapes

```
python shapes_simulate.py
```

#### Correspond the shapes

```
python shapes_learn_correspondence.py
```

#### Visualize correspondence

```
python shapes_visualize.py
```

### Walker Experiments

Coming soon.