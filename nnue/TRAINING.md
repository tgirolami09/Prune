# how currently is trained the nnue

## the layers :
- 768 inputs, 12 for each squares (each combinaison of type+color)
- one 64 neuron hidden layer

## one training step :
- generating data :
    - playing games against itself
    - startpos are position from the UHO book
    - when a mate is seen, the game is adjujed
    - store \<fen\> | \<search score\> | \<static score\> | \<best move\> | \<result of the game\> in files
- training on data using wdl as target
    - don't use data when:
        1. |search score| > 1500cp
        2. |static score| > 1200cp
        3. best move is a capture
        4. current position in check

## whole training :
- start with a nnue that simulate a material evaluation
- then, the new nn is better than the lastest :
    - make one training step by using the lastest nn