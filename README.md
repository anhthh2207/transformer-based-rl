# Transformer-based Reinforcement Learning

Final Project COMP2050 - VinUniversity

**Team**: Tran Quoc Bao, Tran Huy Hoang Anh, Le Chi Cuong

**Description**: We reimplement three SOTA reinforcement learning models including Decision Transformer, Trajectory Transformer, and Conservative Q-Learning. Then, they are tested on different games on the Atari Gymnasium environment to compare their performance.

## How to run

### Setup Environment

Create new environment
```bash
conda env create -f environment.yml
conda activate transformer-based-rl
```
Install atari:
```bash
pip install gymnasium[atari]
pip install autorom[accept-rom-license]
```
Useful links for troubleshooting: [1](https://stackoverflow.com/questions/63080326/could-not-find-module-atari-py-ale-interface-ale-c-dll-or-one-of-its-dependenc) 
[2](https://stackoverflow.com/questions/69442971/error-in-importing-environment-openai-gym)

### Download Dataset
Import ROMS following [this instruction](https://github.com/openai/atari-py#roms)
```bash
cd data
pip install git+https://github.com/takuseno/d4rl-atari
python download_dataset.py
```

### Run Experiment
```bash
python experiments.py --game [GAME] --dataset [DATA_TYPE] --model_type [MODEL]
```

**Available games**: boxing, casino, alien, adventure, breakout

**Data types**: mixed, medium, expert

**Model types**: conservative_q_learning, decision_transformer, trajectory_transformer

**Example**:
```bash
python experiments.py --game casino --dataset medium --model_type decision_transformer
