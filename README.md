# Transformer-based Reinforcement Learning

Final Project COMP2050 - VinUniversity

**Team**: Tran Quoc Bao, Tran Huy Hoang Anh, Le Chi Cuong

**Description**: 

- Goal 1: Implement a minimal version of the Decision Transformer model to play the Atari Breakout game

- Goal 2: Train the model on multiple environments and test its ability to generalize to new distributions

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
python download_dataset.py --mix_games False
```
Use `--mix_games True` to download synthetic dataset used for the distribution shift experiment

<!-- ### Run Experiment
```bash
python experiments.py --game [GAME] --dataset [DATA_TYPE] --model_type [MODEL]
```

**Game options**: `boxing`, `alien`, `breakout` -->

**Data options**: `mixed`, `medium`, `expert`

- `mixed` denotes datasets collected at the first 1M steps.
- `medium` denotes datasets collected at between 9M steps and 10M steps.
- `expert` denotes datasets collected at the last 1M steps.

<!-- **Example**:
```bash
python experiments.py --game boxing --dataset mixed --model_type decision_transformer
``` -->