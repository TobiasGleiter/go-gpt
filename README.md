# 🤖 Build GPT from scratch in Go.

This project follows the tutorial from Andrej Karpathy on [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
Let's have a look how far I could come...

## Todos (from the Video)

- [x] Reading and exploring the data
- [x] Tokenization, train/val split
- [x] Data loader: batches of chunks of data
- [x] Simplest baseline: bigram language model, loss, generation
- [x] Train the bigram model (not sure if the model is correctly trained)
- [ ] Version 1: Averaging past context with for loop, the weakes form of...
- [ ] The trick in self-attention: matrix multiply as weighted aggregation
- [ ] Version 2: Using matrix multiply
- [ ] Version 3: Adding softmax
- [ ] Minor code cleanup
- [ ] Positional Encoding
- [ ] Version 4: Self-attention
- [ ] Note 1: Attention as communication
- [ ] Note 2: Attention has no notion of space, operates over sets
- [ ] Note 3: There is no communication across batch dimension
- [ ] Note 4: Encoder blocks vs. decoder blocks
- [ ] Note 5: Attention vs self-attention vs cross-attention
- [ ] Note 6: "Scaled" self-attention. Why divide by sqrt(head_size)
- [ ] Inserting a single self-attention block to our network
- [ ] Multi-headed self-attention
- [ ] Feedforward layers of transformer architecture
- [ ] Residual connections
- [ ] Layernorm (and its relationship to our previous batchnorm)
- [ ] Scaling up the model! Creating a few variables. Adding dropout
- [ ] Encoder vs decoder vs both ?
