#include "sydutil.hpp"

void node::write(vector<uint8_t>& buffer) const {
    uint8_t s = childs.size();
    buffer.push_back(s);
    mvscore.dump(buffer);
    for (const node& a : childs) {
        a.write(buffer);
    }
}

void node::writeroot(vector<uint8_t>& buffer, const GameState& root) const {
    dumpposition(buffer, root);
    uint8_t s = childs.size();
    mvscore.dump(buffer);
    buffer.push_back(s);
    for (const node& a : childs) {
        a.write(buffer);
    }
}