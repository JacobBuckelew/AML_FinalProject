# Advanced Machine Learning Final Project

## Running Experiments

It's recommended to setup a new conda environment with python 3.11. 


1. Install necessary packages listed in `requirements.txt`:

    ```shell
    pip3 install -r requirements.txt
    cd src
    ```

2. Ensure the ROOT Path located at the top of the scripts `load_data.py` , `train_CANF.py` , `test_CANF.py`, `train_ganf.py`, and `test_ganf.py` are set correctly. 

3. Run the shell files to train and test each model:

    ```shell
    ./train.sh
    ```

There should be 3 training files:

    - `train.sh`: trains our approach (CANF/CNF)

    - `train_nocontext.sh`: trains basic normalizing flows (NF)

    - `train_ganf.sh`: trains GANF

And 3 test files:

    - `test.sh`: tests our approach

    - `test_nocontext.sh`: tests NF

    - `test_ganf.sh`: tests GANF

4. To access results see the Log density images in `figures/` and the `log/test/` directory for each model's AUC score.
