<h1 align="center">COMP_SCI 396 Deep Learning SP2022</h1>
<hr/>

---

This repository holds notebooks and code for the course COMP_SCI 396 Deep Learning, Spring 2021, Northwestern University.


### Getting Started

If you want to run the notebooks locally rather than through Google Colab, the following steps are recommended:

1. Clone this repository from GitHub:
```bash
git clone https://github.com/interactiveaudiolab/course-deep-learning.git && cd course-deep-learning
```

2. Download and install Miniconda, a Python virtual environment manager

3. Create a new virtual environment using the following command:
```bash
conda create --name course-deep-learning python=3.8
```

4. Activate the new environment:
```bash
conda activate course-deep-learning
```

5. Install the required packages (you can then skip the installation instructions at the beginning of each notebook):
```bash
pip install -r requirements.txt
```

6. Run the following command to ensure your new virtual environment is visible to Jupyter:
```bash
python -m ipykernel install --user --name=course-deep-learning
```

7. When PyTorch downloads a dataset and tries to create a progress bar, Jupyter can occasionally cause issues. To avoid this, run:
```bash
conda install -c conda-forge ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Notebooks

TODO: put Google Colab links here once notebooks are finalized
