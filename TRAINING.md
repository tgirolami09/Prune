# how currently is trained the nnue

## the layers :
- 768 inputs, 12 for each squares (each combinaison of type+color)
- one 64 neuron hidden layer
- 8 output buckets

## one training step :
- generating data :
    - gen ~100k games for the first iterations
    - playing games against itself
    - startpos are position from the UHO book (will try to change to 8 random moves)
    - use viriformat
- training on data using wdl=0.75 as target
    - use the default filtrer provided by viriformat

## whole training :
- start with a random nnue, available by :
```bash
./prune "setoption name nnueFile value random"
```
(where you will have prune using the exact same random net)
- then, the new nn is better than the latest :
    - make one training step by using the latest nn
- otherwise, augmenting the number of games of datagen