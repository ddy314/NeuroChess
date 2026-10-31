// Direct minimax/anchor/proof invariants, independent of a model's playing skill.
#include "../neurochess/native/best_first.cpp"
#include <cassert>
#include <iostream>
#include <random>
int main(int argc,char **argv) {
  assert(argc==2);
  bf::Engine e(argv[1],{16,4096,16,8,4,2,1000});
  e.nodes.resize(5);
  e.nodes[0].expanded=true; e.nodes[0].children={1,2};
  e.nodes[1].parent=0; e.nodes[2].parent=0;
  e.nodes[1].expanded=true; e.nodes[1].children={3,4};
  e.nodes[3].parent=1; e.nodes[4].parent=1;
  e.nodes[3].value=.7f; e.nodes[4].value=-.2f; e.nodes[2].value=.4f;
  e.backup(1);
  // Negamax: node 1 chooses -.(-.2)=+.2; root chooses -.2 over -.4.
  assert(std::abs(e.nodes[1].value-.2f)<1e-6);
  assert(std::abs(e.nodes[0].value+.2f)<1e-6);
  // A correction survives descendant backup, instead of disappearing on expansion.
  e.nodes[1].anchor=-.8f; e.nodes[1].anchor_weight=.75f; e.backup(1);
  assert(std::abs(e.nodes[1].value-(-.55f))<1e-6);
  assert(std::abs(e.nodes[0].value-.55f)<1e-6);
  assert(e.nodes[0].solved==bf::Unknown);
  // Exact outcomes override every approximation, including the neural anchor.
  e.nodes[3].solved=-1; e.nodes[3].value=-1; e.nodes[3].frozen=true;
  e.backup(1);
  assert(e.nodes[1].solved==1 && e.nodes[1].value==1);
  assert(e.nodes[0].solved==bf::Unknown); // The other root move remains unknown.
  e.nodes[2].solved=1; e.nodes[2].value=1; e.nodes[2].frozen=true;
  e.backup(0); assert(e.nodes[0].solved==-1 && e.nodes[0].value==-1);
  bf::Node guarded;guarded.anchor=-.9f;guarded.anchor_weight=.75f;e.correction_clip=.15f;
  assert(std::abs(e.fuse(guarded,.6f)-.4875f)<1e-6);
  assert(guarded.anchor_clipped);
  e.correction_clip=0;e.cpu_correction_clip=.15f;
  guarded.cpu_ready=true;assert(std::abs(e.fuse(guarded,.6f)-.4875f)<1e-6);
  guarded.cpu_ready=false;assert(std::abs(e.fuse(guarded,.6f)-(-.525f))<1e-6);
  bf::Engine ranker(argv[1],{16,4096,16,8,4,0,0});
  ranker.nodes.resize(3);ranker.nodes[0].children={1,2};
  ranker.nodes[1].cheap=-.6f;ranker.nodes[1].value=.8f;
  ranker.nodes[2].cheap=-.4f;ranker.nodes[2].value=-.2f;
  assert(ranker.rank_root()[0]==2);
  ranker.root_value_clip=.15f;assert(ranker.rank_root()[0]==1);
  ranker.nodes[2].solved=-1;assert(ranker.rank_root()[0]==2);
  bf::Tactical proof(e.weights);
  chess::Board b("rq2r1k1/3Q1p1p/1p3p2/p7/6R1/4PN2/PPP3PP/2K1R3 b - - 1 18");
  b.makeMove(chess::uci::uciToMove(b,"g8h8"));
  proof.mate_limit=1;assert(!proof.mate(b,b.sideToMove(),5,false));assert(proof.mate_nodes<=1);
  proof.mate_nodes=0;proof.mate_limit=16384;assert(proof.mate(b,b.sideToMove(),5,false));
  assert(proof.mate_nodes<=16384);
  std::mt19937 rng(7101);
  for(int game=0;game<5;++game) {
    chess::Board position;
    for(int ply=0;ply<80;++ply) {
      chess::Movelist moves;chess::movegen::legalmoves(moves,position);
      if(moves.empty())break;
      auto before_hash=position.hash();
      for(auto item:bf::Tactical::order(position,moves)) {
        position.makeMove(item.move);assert(item.check==position.inCheck());position.unmakeMove(item.move);
        assert(position.hash()==before_hash);
      }
      position.makeMove(moves[rng()%moves.size()]);
    }
  }
  for(const char *fen:{"r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
      "7k/8/8/3pP3/8/8/8/7K w - d6 0 1","7k/P7/8/8/8/8/8/7K w - - 0 1"}) {
    chess::Board position(fen);chess::Movelist moves;chess::movegen::legalmoves(moves,position);
    for(auto item:bf::Tactical::order(position,moves)) {
      position.makeMove(item.move);assert(item.check==position.inCheck());position.unmakeMove(item.move);
    }
  }
  bf::Engine trail(argv[1],{16,4096,16,8,4,0,0});
  trail.start(chess::constants::STARTPOS,"g1f3 g8f6 f3g1 f6g8 e2e4");
  assert(!trail.nodes[0].repeated);
  trail.start(chess::constants::STARTPOS,"g1f3 g8f6 f3g1 f6g8 e2e4 b8c6 b1c3 c6b8 c3b1");
  assert(trail.nodes[0].repeated);
  std::cout << "minimax signs, persistent anchors and exact proof boundaries passed\n";
}
