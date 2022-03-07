# story-template-extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project attempts to extract a set of narrative arc templates from albums in the [Free Music Archive dataset](https://github.com/mdeff/fma).

## Installation

This project is implemented in [Python](https://www.python.org/). To use this project, first install the required packages using pip:
```bash
pip install -r requirements.txt
```

Afterwards, you can learn a series of templates by using the `build.sh` file to construct the set of Slurm Workload Manager tasks for execution on a cluster:
```bash
sbatch --array=0-$(./build.sh) ./cscs.sh
```
