General info:

Input to each transformer block of size (B,T,C):

- B: batch size
- T: input sequence length (later be equal to 3*h_dim)
- C: vector state (state_dim) + action (action_dim = 4)

Input to the model: 

- timesteps: matrix of size (B,T)
- states: matrix of size (B,T*state_dim)
- actions: matrix of size (B,T*action_dim)
- returns_to_go: matrix of size (B,T)

Notice: T = context_len $\Rightarrow$ Later stack 3 matrices (after embedding) and get sequence length = 3\*T $\Rightarrow$ input size (B,3\*T,h_dim)

Output: (state_preds (state_dim), action_preds (action_dim), return_preds (1))