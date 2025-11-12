<div align="center">
    <h1>Prune</h1>
    <img src="forest4.jpeg" alt="drawing" width="300"/>
</div>

this is a hobby project made by two passionate about informatic.

currently, the engine includes:
- Evaluation:
    - nnue
    - trained on self-gen data with own trainer
- Search:
    - iterative Deepening
    - quiescence search
    - aspiration window
    - enhanced forward pruning
    - Reduction:
        - late move reduction
    - pruning:
        - null move pruning
        - reverse futility pruning
        - razoring at depth 1
    -  transposition table
    - move ordering:
        - killer move
        - PV move
        - SEE ordering for depth > 5
        - MVV-LVA otherwise
        - root move ordering based on node count
        - history heuristic
- UCI protocol:
    - fancy commands:
        - version (give the commit's hash when it was compiled)
        - arch (give the arch used to compile)
        - bench (test of 48 positions, which give the node count/depth sum to compare search, also give mean nps)
        - runQ (run quiescnce search on the given position)
        - d (print the current position [after the moves])
        - position kiwipete (set kiwipete's position)
- does not support multithreading yet

to compile, just run 

```bash
make prune
```

in the core/ directory, and you will have the executable prune, which contains the engine (so it is moveable)