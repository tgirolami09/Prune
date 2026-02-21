<div align="center">
    <h1>Prune</h1>
    <img src="forest4.jpeg" alt="drawing" width="300"/>
</div>

this is a hobby project made by two passionate about informatic.

currently, the engine includes:
- Evaluation:
    - nnue
        - 1024 HL
        - 8 output buckets
        - horizontal mirroring
        - 10 iterations from random net
    - trained on self-gen data using [bullet](https://github.com/jw1912/bullet) (if you want the data used to train it, you can directly ask one of us)
    - correction history :
        - pawn
        - previous move
        - continuation move
    - material scaling
- Search:
    - iterative Deepening
    - quiescence search
    - aspiration window
    - enhanced forward pruning
    - Reduction:
        - late move reduction :
            - log base
            - history
            - tactical moves
    - pruning:
        - null move pruning
        - reverse futility pruning (improving and not improving)
        - razoring at depth 1
        - late move pruning
        - history pruning
        - capture history pruning
        - futility pruning
        - SEE pruning in QS
    - transposition table
        - buckets of 3 entries
    - move ordering:
        - killer move
        - PV move
        - SEE
        - MVV-LVA
        - history heuristic
        - capthist
    - singular extension :
        - singular
        - double
        - negative
        - cutnode negext
    - IIR
- most of the search parameters has been tuned over 2k iterations of 16 40+0.4 games
- movegen
    - use bitboards
    - fully legal movegen
- UCI protocol:
    - fancy commands:
        - version (give the commit's hash when it was compiled)
        - arch (give the arch used to compile)
        - bench (test of 48 positions, which give the node count/depth sum to compare search, also give average nps)
        - runQ (run quiescnce search on the given position)
        - print (print the current position [after the moves])
        - position kiwipete (set kiwipete's position)
- option :
    - nnueFile
        - for bullet format quantised
        - random will init a random net
        - embed will use the embeded one
    the other ones should be known, refer to the stockfish readme
- support multithreading (lazy smp)
- don't support yet frc, planed for v4

to compile, just run 

```bash
make prune
```

in the core/ directory, and you will have the executable prune, which contains the engine (so it is moveable)

## note 
for the version naming, we currently use the following :
- 3rd digit is only for bug fix versions
- 2nd digit is for elo improvement (for now, we target a +190 elo per release)
- 1st digit is for new features (as smp for v3, or planed frc for v4)

## thanks
- stockfish discord :
    1. TM improvement (took some constant from heimdall for init)
    2. corrhist help  (took some constant from sirius for update)
    3. speedup of the nnue (lizard updates)
- chessprogramming wiki
    - warn for new dev, can be outdated, so don't forget to sprt every changes
- especially to :
    - swedishchef (the dev from pawnocchio and vine) for helping me debugging the viriformat writer, and giving me some search ideas