General info:

Input to the model: 

- timesteps: matrix of size (B,T)
- states: matrix of size (B,T,state_dim,state_dim)
- actions: matrix of size (B,T*action_dim)
- returns_to_go: matrix of size (B,T)

For atari, state_dim = 84

Notice: T = context_len $\Rightarrow$ Later stack 3 matrices (after embedding) and get sequence length = 3\*T $\Rightarrow$ input size (B,3\*T,h_dim)

Output: (state_preds (state_dim), action_preds (action_dim), return_preds (1))