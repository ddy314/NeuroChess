// Direct invariants for the same implementation compiled into libmcts.so.
#include "../neurochess/native/mcts.cpp"
#include <cassert>
#include <iostream>
uint64_t perft(Board &b, int depth) {
  if (!depth)
    return 1;
  Movelist moves;
  movegen::legalmoves(moves, b);
  if (depth == 1)
    return moves.size();
  uint64_t n = 0;
  for (auto m : moves) {
    b.makeMove(m);
    n += perft(b, depth - 1);
    b.unmakeMove(m);
  }
  return n;
}
int main() {
  Config c{64, 16, 6, 12, 128, 1.5f, .2f, 1.5f, .75f};
  Engine engine(c);
  Tree t(constants::STARTPOS);
  int child = t.node(), leaf = t.node();
  t.nodes[0].begin = 0;
  t.nodes[0].count = 1;
  t.nodes[child].begin = 1;
  t.nodes[child].count = 1;
  Edge a, b;
  a.child = child;
  b.child = leaf;
  t.edges = {a, b};
  Job job{&t, leaf, {0, 1}};
  engine.backup(job, .7f);
  assert(std::abs(t.edges[0].q() - .7f) < 1e-6);
  assert(std::abs(t.edges[1].q() + .7f) < 1e-6);
  assert(t.nodes[0].visits == 1 && t.nodes[child].visits == 1 &&
         t.nodes[leaf].visits == 1);
  // Unknown, pruned alternatives prevent false loss/draw proofs.
  t.nodes[0].count = 2;
  t.nodes[0].solved = Unknown;
  t.edges[0].child = child;
  t.nodes[child].solved = 1;
  t.edges[1].child = -1;
  t.edges[1].pruned = true;
  engine.proof(t, 0);
  assert(t.nodes[0].solved == Unknown);
  t.edges[1].child = leaf;
  t.nodes[leaf].solved = 0;
  engine.proof(t, 0);
  assert(t.nodes[0].solved == 0);
  t.nodes[leaf].solved = -1;
  engine.proof(t, 0);
  assert(t.nodes[0].solved == 1);
  // Time-limited search may have a high simulation safety cap. Prune by time
  // fraction rather than waiting to consume75% of that unreachable cap.
  Engine timed(c);Tree timed_tree(constants::STARTPOS);
  timed_tree.nodes[0].count=20;timed_tree.edges.resize(20);
  timed.time_fraction=.5f;timed.prune(timed_tree);
  assert(!timed_tree.nodes[0].pruned);
  timed.time_fraction=.8f;timed.prune(timed_tree);
  assert(timed_tree.nodes[0].pruned && timed.stats.pruned==8);
  Tree waiting(constants::STARTPOS);waiting.nodes[0].count=2;
  waiting.edges.resize(2);int loss=waiting.node(),pending=waiting.node();
  waiting.edges[0].child=loss;waiting.nodes[loss].solved=1;
  waiting.edges[1].child=pending;waiting.nodes[pending].pending=true;
  assert(engine.select(waiting,0)==0); // Historical incorrect loss fallback.
  engine.pending_loss_guard=true;
  assert(engine.select(waiting,0)==-1); // Flush/wait for viable pending child.
  waiting.nodes[pending].pending=false;
  assert(engine.select(waiting,0)==1);
  Board start;
  assert(perft(start, 4) == 197281);
  Board kiwi(
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
  assert(perft(kiwi, 3) == 97862);
  std::cout << "backup, exact proofs and perft passed\n";
}
