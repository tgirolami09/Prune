using namespace std;


//Class to generate legal moves
class LegalMoveGenerator{
    //Transforms a bitboard of valid end positions into a list of the corresponding moves
    vector<Move> maskToMoves(int start, big mask){
        vector<Move> res;
        while(mask){
            int bit = ffsll(mask);
            mask ^= 1 << bit;
            //Need to add logic for pawn promotion
            res.push_back({start,bit,-1});
        }
        return res;
    }

    //Returns all allowed spaces for a piece to move
    //If the king is not in check then everywhere
    //Else only moves preventing check
    vector<big> kingInCheck(GameState state){
        //Look at the squares the enemy attacks
        return {};
    }

    vector<Move> kingMoves(GameState state){
        return {};
    }

    vector<Move> pawnMoves(GameState state, big moveMask, big captureMask){
        return {};
    }

    vector<Move> knightMoves(GameState state, big moveMask, big captureMask){
        return {};
    }

    vector<Move> slidingMoves(GameState state, big moveMask, big captureMask){
        return {};
    }

    //Returns all legal moves for a position
    public : vector<Move> generateLegalMoves(GameState state){
        //All allowed spots for a piece to move (not allowed if king is in check)
        big moveMask = -1; //Totaly true
        //All allowed spots for a piece to capture another one (not allowed if there is a checker)
        big captureMask = -1; //Totaly true

        vector<big> currentMasks = kingInCheck(state);
        moveMask = currentMasks[0];
        captureMask = currentMasks[1];

        vector<Move> legalMoves;

        vector<Move> legalKingMoves = kingMoves(state);
        vector<Move> legalPawnMoves = pawnMoves(state,moveMask,captureMask);
        vector<Move> legalKnightMoves = knightMoves(state,moveMask,captureMask);
        vector<Move> legalSlidingMoves = slidingMoves(state,moveMask,captureMask);

        legalMoves.insert(legalMoves.end(),legalKingMoves.begin(),legalKingMoves.end());
        legalMoves.insert(legalMoves.end(),legalPawnMoves.begin(),legalPawnMoves.end());
        legalMoves.insert(legalMoves.end(),legalKnightMoves.begin(),legalKnightMoves.end());
        legalMoves.insert(legalMoves.end(),legalSlidingMoves.begin(),legalSlidingMoves.end());

        return legalMoves;
    }
};
