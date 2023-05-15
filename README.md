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
In order to use atari, you must import ROMS following [this instruction](https://github.com/openai/atari-py#roms)

### Download Dataset
```bash
cd data
pip install git+https://github.com/takuseno/d4rl-atari
python download_dataset.py
```

### Run Experiment
```bash
python experiments.py --game [GAME] --dataset [DATA_TYPE] --model_type [MODEL]
```

**Available games**: boxing, asterix, alien, adventure, breakout

**Data types**: mixed, medium, expert

**Model types**: conservative_q_learning, decision_transformer, trajectory_transformer

**Example**:
```bash
python experiments.py --game boxing --dataset medium --model_type decision_transformer
