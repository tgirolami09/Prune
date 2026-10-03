#include "sydutil.hpp"
#include <memory>
#include "Const.hpp"
#include "GameState.hpp"
#include "LegalMoveGenerator.hpp"
#include "TranspositionTable.hpp"

void node::dumpinfo(vector<uint8_t>& buffer) const {
    mvscore.dump(buffer);
    buffer.push_back(min(depth / fracDepth, 256 / 3) * 3 + bound);
}
void node::write(vector<uint8_t>& buffer, const vector<vector<node>>& nodetable) const {
    dumpinfo(buffer);
    buffer.push_back(childs.size());
    for (const auto& p : childs) {
        const node& a = nodetable[p.hashidx][p.bucketidx];
        dumpmove(p.mv, buffer);
        a.write(buffer, nodetable);
    }
}

void node::writeroot(vector<uint8_t>& buffer, const GameState& root,
                     const vector<vector<node>>& nodetable) const {
    dumpposition(buffer, root);
    dumpinfo(buffer);
    buffer.push_back(childs.size());
    for (const auto& p : childs) {
        const node& a = nodetable[p.hashidx][p.bucketidx];
        dumpmove(p.mv, buffer);
        a.write(buffer, nodetable);
    }
}

void node::update(const infoScore& ttentry) {
    if ((ttentry.typeNode() == EXACT) ^ (bound == EXACT)
            ?  // one of them is exact
            ttentry.typeNode() == EXACT
            : (((bool)ttentry.bestMove ^ (bool)mvscore.move)
                   ?  // only one of them has a bestmove
                   (bool)ttentry.bestMove
                   : ttentry.depth >= depth  // last condition, keep the
                                             // entry with biggest depth
               )) {
        bound = ttentry.typeNode();
        mvscore.move = ttentry.bestMove;
        mvscore.score = ttentry.score;
        depth = ttentry.depth;
    }
}

void totree(GameState& state, const transpositionTable& tt, Link curnode,
            vector<vector<node>>& nodetable, int curage, int& nbNew, const vector<Move>& legalMoves,
            const int nbMoves) {
    unique_ptr<PositionSnapshot> snap = make_unique<PositionSnapshot>();
    snap->save(state);
    LegalMoveGenerator generator;
    vector<Move> nextlegalMoves(maxMoves);
    bool inCheck;
    big dangerPositions;
    for (int i = 0; i < nbMoves; i++) {
        state.playMove(legalMoves[i]);
        bool ttHit = false;
        const auto& ttentry = tt.getEntry(state, ttHit);
        // printf("%s (%s) => %d\n", state.toFen().c_str(), legalMoves[i].to_str().c_str(), ttHit);
        if (ttHit && ttentry.depth > fdepth<5>) {
            auto [idx, rem] = getIndex(state, nodetable.size());
            generator.initDangers(state);
            int nextnbMoves =
                generator.generateLegalMoves(state, inCheck, &nextlegalMoves[0], dangerPositions);
            bool isin = !ttentry.bestMove;
            for (int idM = 0; idM < nbMoves; idM++) {
                const auto& mv = nextlegalMoves[idM];
                if (mv == ttentry.bestMove) {
                    isin = true;
                    break;
                }
            }
            if (isin) {
                assert(idx < nodetable.size());
                bool added = false;
                for (uint32_t bucketidx = 0; bucketidx < nodetable[idx].size(); bucketidx++) {
                    auto& nnode = nodetable[idx][bucketidx];
                    if (nnode.hash == state.zobristHash) {
                        nnode.update(ttentry);
                        if (nnode.age != curage) {
                            nnode.age = curage;
                            Link l{(uint32_t)idx, bucketidx, nullMove};
                            totree(state, tt, l, nodetable, curage, nbNew, nextlegalMoves,
                                   nextnbMoves);
                        }
                        added = true;
                    }
                }
                if (!added) {
                    nbNew++;
                    // printf("add position %s %s\n", state.toFen().c_str(),
                    // legalMoves[i].to_str().c_str());
                    node newnode(ttentry, state.zobristHash);
                    newnode.age = curage;
                    Link newlink{(uint32_t)idx, (uint32_t)nodetable[idx].size(), legalMoves[i]};

                    nodetable[curnode.hashidx][curnode.bucketidx].childs.push_back(newlink);
                    nodetable[idx].push_back(newnode);
                    totree(state, tt, newlink, nodetable, curage, nbNew, nextlegalMoves,
                           nextnbMoves);
                }
            }
        }
        snap->restore(state);
    }
}