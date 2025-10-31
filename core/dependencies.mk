Const.o: Const.cpp Const.hpp
Const.hpp:
Evaluator.o: Evaluator.cpp Evaluator.hpp Const.hpp GameState.hpp Move.hpp \
 Functions.hpp LegalMoveGenerator.hpp NNUE.hpp
Evaluator.hpp:
Const.hpp:
GameState.hpp:
Move.hpp:
Functions.hpp:
LegalMoveGenerator.hpp:
NNUE.hpp:
LegalMoveGenerator.o: LegalMoveGenerator.cpp LegalMoveGenerator.hpp \
 Const.hpp Functions.hpp GameState.hpp Move.hpp
LegalMoveGenerator.hpp:
Const.hpp:
Functions.hpp:
GameState.hpp:
Move.hpp:
NNUE.o: NNUE.cpp NNUE.hpp Const.hpp Functions.hpp
NNUE.hpp:
Const.hpp:
Functions.hpp:
TranspositionTable.o: TranspositionTable.cpp TranspositionTable.hpp \
 Const.hpp GameState.hpp Move.hpp Functions.hpp
TranspositionTable.hpp:
Const.hpp:
GameState.hpp:
Move.hpp:
Functions.hpp:
engine.o: engine.cpp Const.hpp Move.hpp Functions.hpp GameState.hpp \
 BestMoveFinder.hpp TranspositionTable.hpp Evaluator.hpp \
 LegalMoveGenerator.hpp NNUE.hpp MoveOrdering.hpp loadpolyglot.hpp \
 polyglotHash.hpp
Const.hpp:
Move.hpp:
Functions.hpp:
GameState.hpp:
BestMoveFinder.hpp:
TranspositionTable.hpp:
Evaluator.hpp:
LegalMoveGenerator.hpp:
NNUE.hpp:
MoveOrdering.hpp:
loadpolyglot.hpp:
polyglotHash.hpp:
