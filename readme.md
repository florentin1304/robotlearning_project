# Starting code for course project of Robot Learning - 01HFNOV

Official assignment at [Google Doc](https://docs.google.com/document/d/1wV60fIh1jCifi9O4ID6R61IgQry79ftxOuPJj9JyCow/edit?usp=sharing)


## Getting started

Before starting to implement your own code, make sure to:
1. read and study the material provided (see Section 1 of the assignment)
2. read the documentation of the main packages you will be using ([mujoco-py](https://github.com/openai/mujoco-py), [Gym](https://github.com/openai/gym), [stable-baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html))
3. play around with the code in the template to familiarize with all the tools. Especially with the `test_random_policy.py` script.


### 1. Local (Linux)

if you have a Linux system, you can work on the course project directly on your local machine. By doing so, you will also be able to render the Mujoco Hopper environment and visualize what is happening. This code has been tested on Linux with python 3.7.

**Dependencies**
- Install MuJoCo and the Python Mujoco interface following the instructions here: https://github.com/openai/mujoco-py
- Run `pip install -r requirements.txt` to further install `gym` and `stable-baselines3`.

Check your installation by launching `python test_random_policy.py`.


### 2. Local (Windows)
As the latest version of `mujoco-py` is not compatible for Windows explicitly, you may:
- Try installing WSL2 (requires fewer resources) or a full Virtual Machine to run Linux on Windows. Then you can follow the instructions above for Linux.
- (not recommended) Try downloading a [previous version](https://github.com/openai/mujoco-py/blob/9ea9bb000d6b8551b99f9aa440862e0c7f7b4191/) of `mujoco-py`.
- (not recommended) Stick to the Google Colab template (see below), which runs on the browser regardless of the operating system. This option, however, will not allow you to render the environment in an interactive window for debugging purposes.


### 3. Google Colab

You can also run the code on [Google Colab](https://colab.research.google.com/)

- Download all files contained in the `colab_template` folder in this repo.
- Load the `test_random_policy.ipynb` file on [https://colab.research.google.com/](colab) and follow the instructions on it

NOTE 1: rendering is currently **not** officially supported on Colab, making it hard to see the simulator in action. We recommend that each group manages to play around with the visual interface of the simulator at least once, to best understand what is going on with the underlying Hopper environment.

NOTE 2: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.