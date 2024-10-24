
# Tom Brady Superbowl Success

This repository contains an analysis of Tom Brady's Superbowl success using Python. This guide will help you set up the environment and install all dependencies required to run the project.

## Prerequisites

Before starting, ensure you have the following installed on your system:

- **Git**: Used for cloning the repository.
- **Homebrew**: Package manager for macOS, to install Python 3.11.
- **Python 3.11**: Required for running the project.
- **pip**: Python package installer, comes with Python 3.11.

If you don't have Homebrew installed, follow the instructions at [https://brew.sh/](https://brew.sh/) to install it.

## Installation Steps

### 1. Clone the repository

First, clone this repository to your local machine using Git:

```bash
git clone https://github.com/shashwatmaharjan/tom-brady-superbowl-success.git
cd tom-brady-superbowl-success
```

### 2. Install Python 3.11 using Homebrew

If Python 3.11 is not installed on your system, you can use Homebrew to install it:

```bash
brew install python@3.11
```

After installing, ensure Python 3.11 is the default version by running:

```bash
brew link python@3.11
```

Verify the installation:

```bash
python3 --version
# It should display Python 3.11.x
```

### 3. Create a virtual environment (optional but recommended)

To avoid conflicts with other Python packages, it's recommended to create a virtual environment for the project. Run the following commands:

```bash
python3 -m venv venv
source venv/bin/activate
```

If the environment is activated, you should see `(venv)` in your terminal prompt.

### 4. Install project dependencies

Now, install all the required dependencies using `pip` and the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Running the project

Once the dependencies are installed, you're all set to run the project. Execute the Python scripts or start the analysis as needed.

### 6. Deactivating the virtual environment

When you're done, you can deactivate the virtual environment by running:

```bash
deactivate
```

## Contributing

Feel free to open issues or contribute to the project by submitting a pull request. Make sure to follow the contributing guidelines.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
