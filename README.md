## eGAIT: Multi-Skilled Policy for Energy-efficient Gait Transitions

Further development (new features, bug fixes, etc.) occurs in the `master` branch.

The `paper` branch contains the original code accompanying the paper:

eGAIT: Multi-Skilled Policy for Energy-efficient Gait Transitions
    by Anonymous Authors

This project provides an environment for executing multi-skill policy switches,
adaptable to real-time velocity changes.

Project page:  https://sdsreview.github.io/egait_review/

---

## Getting Started

Recommended setup:
- Python 3.9
- Ubuntu 20.04 LTS

Install the repository, in a terminal run: 
```bash
git clone https://github.com/sdsreview/egait_review.git
cd egait_review/
```


### 1. Create a Conda Environment

Use the following commands to set up a new Conda environment named `egait`:

```bash
conda create -n egait python=3.9
conda activate egait
pip install -r requirements.txt
```

### 2. Install Optional MPC Extension

If you plan to use the MPC controller, install the optional extension:

```bash
python3 setup.py install --user
```

### 3. Install System Dependencies

Install OpenMPI development headers required for parallelization:

```bash
sudo apt install libopenmpi-dev
```

---

## Download Pretrained Checkpoints

Download the pretrained eGAIT policy checkpoints from:

https://drive.google.com/drive/folders/15neku_yEqlh9RGHGGyWXrtzUGVWKLEYx?usp=sharing

Then move the `high_level` folder into the repository's evaluation script directory:

```bash
mv ~/Downloads/high_level /home/egait/egait_code/egait/evaluation_scripts/
```

---

## Running the Evaluation Script

To run the main evaluation script:

```bash
python egait/evaluation_scripts/egait_evaluator.py
```

You can optionally pass arguments, for example:

```bash
python egait/evaluation_scripts/egait_evaluator.py --use_slider=True --use_plane=False
```

---

## Credits

Parts of this repository build upon tools developed by Google Robotics.
The controller system is maintained in part by Erwin Coumans and was
extended for academic research in adaptive and energy-efficient quadruped
gait transitions.
