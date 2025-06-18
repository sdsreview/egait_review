
```
eGAIT
=====

Further development (new features, bug fixes, etc.) occurs in the 'master' branch.
=======
# eGAIT: Multi-Skilled Policy for Energy-efficient Gait Transitions"

The 'paper' branch of this repository contains the original code accompanying the paper:

    "eGAIT: Multi-Skilled Policy for Energy-efficient Gait Transitions"
    by Anonymous Authors

This work presents an environment for executing multi-skill policy switches,
adaptable to real-time velocity changes.

Project page:
    https://sdsreview.github.io/egait_review/

------------------------------------------------------------------------

Getting Started
===============

Recommended environment:
    - Python 3.9
    - Ubuntu 20.04 LTS

1. Create a Conda Environment
-----------------------------

To set up a new Conda environment named 'egait' using requirements.txt:

    conda create -n egait python=3.9
    conda activate egait
    pip install -r requirements.txt

2. Install Optional MPC Extension
---------------------------------

To install the MPC extension (if needed):

    python3 setup.py install --user

3. Install System Dependencies
------------------------------

Install OpenMPI development headers:

    sudo apt install libopenmpi-dev

------------------------------------------------------------------------

Download Pretrained Checkpoints
===============================

Download the pretrained policy checkpoints from:

    https://drive.google.com/drive/folders/15neku_yEqlh9RGHGGyWXrtzUGVWKLEYx?usp=sharing

Then move the `high_level` folder into the repository's evaluation scripts directory:

    mv ~/Downloads/high_level /home/egait/egait_code/egait/evaluation_scripts/

------------------------------------------------------------------------

Running the Evaluation Script
=============================

To run the eGAIT evaluation:

    python egait/evaluation_scripts/egait_evaluator.py

Optional arguments:

    python egait/evaluation_scripts/egait_evaluator.py --use_slider=True --use_plane=False

------------------------------------------------------------------------

Credits
=======

Parts of this repository build upon tools developed by Google Robotics.
The base controller system is maintained in part by Erwin Coumans, and
was extended for research on adaptive and energy-efficient quadruped
gait transitions.

------------------------------------------------------------------------
```

Let me know if you'd like it saved to a file or added directly to your repo.
