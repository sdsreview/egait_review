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
```


###  Create a Conda Environment

Use the following commands to set up a new Conda environment named `egait`:

```bash
cd egait_review/egait_code/
conda env create -f environment.yml
conda activate egait
```

This step might take a few minuites.


## Download Pretrained Checkpoints

Download the pretrained eGAIT policy checkpoints from:

https://drive.google.com/drive/folders/1rY5rzxWZWbVclIOgcqkLmA8_imL5ZqrD?usp=sharing

Then move the `high_level` folder into the repository's evaluation script directory. In a new terminal:

```bash
mv ~/Downloads/saved_model.zip ~/egait_review/egait_code/egait/evaluation_scripts/
cd /home/egait/egait_code/egait/evaluation_scripts/
unzip saved_model.zip & rm saved_model.zip

```

---

## Running the Evaluation Script

To run the main evaluation script:

```bash
cd ../..
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
