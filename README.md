# Transformer-based Reinforcement Learning

Final Project COMP2050 - VinUniversity

**Team**: Tran Quoc Bao, Tran Huy Hoang Anh, Le Chi Cuong

**Description**: 

- Goal 1: analyze the SOTA transformer-based reinforcement learning models including Decision Transformer and Trajectory Transformer. Another traditional method which is Conservative Q-Learning will also be implemented to compare the performance with the others. The environment used for evaluation is the Atari Gym environment.

- Goal 2: try to combine the use of transformer-based language models in RL problems with a traditional method to solve RL problem which is the actor-critic approach. The combination is born with the hope to increase the stability of the Decision Transformer model.

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

**Game options**: `boxing`, `alien`, `breakout`

**Data options**: `mixed`, `medium`, `expert`

- `mixed` denotes datasets collected at the first 1M steps.
- `medium` denotes datasets collected at between 9M steps and 10M steps.
- `expert` denotes datasets collected at the last 1M steps.


**Model options**: `conservative_q_learning`, `decision_transformer`, `trajectory_transformer` 

- [Conservative Q Learning](conservative-q-learning/README.md)
- [Decision Transformer](decision-transformer/README.md)
- [Trajectory Transformer](trajectory-transformer/README.md)

**Example**:
```bash
python experiments.py --game boxing --dataset mixed --model_type decision_transformer
```