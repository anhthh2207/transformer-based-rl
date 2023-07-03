### Training
``` bash
python online_dt_stacked/train.py --
```

Train with greedy replay buffer
``` bash
python online_dt/train_unstacked.py --buffer_size 256 --gradient_iteration 10 --sample_size 32 --greedy_buffer 1
```

Train with stochastic reply buffer
``` bash
python online_dt/train_unstacked.py --buffer_size 256 --gradient_iteration 10 --sample_size 32 --greedy_buffer 0
```