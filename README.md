# 2 bit Quantum Computer Simulator Readme
To run the simulator, run the following command:

```python simulator.py```

For more information about how it works: [Demo Notebook](demo.ipynb).

## Installing Dependencies from `requirements.txt`

Below are several methods to install the required packages for this project:

### 1. Using pip

The most straightforward way to install dependencies is using `pip`. You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

### 2. Using pip and a virtual environment (with `virtualenv`)

It's often recommended to use a virtual environment to avoid potential conflicts with system packages:

```bash
# First, install virtualenv if it's not installed
pip install virtualenv

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Now install the dependencies
pip install -r requirements.txt
```

### 3. Using conda (for Anaconda or Miniconda users)

If you're using Anaconda or Miniconda, you can create a new environment and install the required packages from `requirements.txt`:

```bash
# Create a new conda environment
conda create --name myenv python=3.8  # Replace 3.8 with your desired Python version

# Activate the environment
conda activate myenv

# Install the dependencies
pip install -r requirements.txt
```

Choose the method that best fits your setup and preferences.