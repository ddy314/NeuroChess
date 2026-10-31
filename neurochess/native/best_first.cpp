// Best-first negamax with cheap incremental evaluation and a separate, hard
// budget for expensive neural queries. No sampling/mean-return backup.
#include "chess_rules.hpp"
#include <array>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <chrono>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
namespace bf {
using namespace chess;
using nc_rules::Unknown;
constexpr int Capacity = 256;
bool repeated_after(const Board &b,bool inherited) {
  // Positions before a pawn move/capture cannot recur on the new reversible
  // trail. Keeping the old flag makes every subsequent node scan legal moves
  // for an impossible next-move repetition claim.
  return b.halfMoveClock()>0 && (inherited || b.isRepetition(1));
}
struct Config {
  int expansions, max_nodes, depth, queries, batch, qdepth, qbudget;
};
struct Accumulator {
  std::array<float, 2048> x; // Only the active 2*width entries are initialized/read.
  float material = 0;        // White material / 600.
};
struct Weights {
  int evaluation_mode=0; // 0 student, 1 material, 2 neutral (causal ablations).
  int width, hidden;
  std::vector<float> bias, features, hidden_w, hidden_b, output;
  float output_b;
  void copy(Accumulator &to,const Accumulator &from) const {
    std::copy_n(from.x.data(),2*width,to.x.data());to.material=from.material;
  }
  static float dot(const float *a,const float *b,int size) {
    float result=0;int j=0;
#if defined(__AVX2__)
    __m256 sum=_mm256_setzero_ps();
    for(;j+8<=size;j+=8)sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_loadu_ps(a+j),_mm256_loadu_ps(b+j)));
    alignas(32) float lanes[8];_mm256_store_ps(lanes,sum);
    for(float v:lanes)result+=v;
#endif
    for(;j<size;++j)result+=a[j]*b[j];
    return result;
  }
  explicit Weights(const char *path) {
    std::ifstream f(path, std::ios::binary);
    char magic[8]; uint32_t w = 0, h = 0;
    f.read(magic, 8); f.read(reinterpret_cast<char *>(&w), 4);
    f.read(reinterpret_cast<char *>(&h), 4);
    if (!f || std::memcmp(magic, "NCHEAP01", 8) || w < 1 || w > 1024 || h < 1 || h > 512)
      throw std::runtime_error("Invalid cheap weight header");
    width = w; hidden = h;
    auto read = [&](std::vector<float> &v, size_t n) {
      v.resize(n); f.read(reinterpret_cast<char *>(v.data()), n*sizeof(float));
      if (!f) throw std::runtime_error("Truncated cheap weights");
      for (float x : v) if (!std::isfinite(x)) throw std::runtime_error("Nonfinite cheap weight");
    };
    read(bias, width); read(features, 768*width);
    read(hidden_w, hidden*(2*width+12)); read(hidden_b, hidden); read(output, hidden);
    f.read(reinterpret_cast<char *>(&output_b), sizeof(float));
    if (!f || !std::isfinite(output_b) || f.peek() != EOF)
      throw std::runtime_error("Invalid cheap weight payload");
  }
  static int feature(int type, int color, int square, int perspective) {
    return (type + (color == perspective ? 0 : 6))*64 + (square ^ (perspective ? 56 : 0));
  }
  void add(Accumulator &a, int type, int color, int square, float sign) const {
    static constexpr float material[] = {100,320,330,500,900,0};
    a.material += sign*material[type]*(color ? -1.f : 1.f)/600.f;
    for (int view = 0; view < 2; ++view) {
      const float *w = &features[feature(type,color,square,view)*width];
      float *x = &a.x[view*width];
      for (int i = 0; i < width; ++i) x[i] += sign*w[i];
    }
  }
  Accumulator initial(const Board &b) const {
    Accumulator a;
    // Root accumulators may be returned/copied as complete objects. Initialize
    // their tail too; hot child clones copy/read only the active entries.
    a.x.fill(0.f);
    for (int view = 0; view < 2; ++view)
      std::copy(bias.begin(), bias.end(), a.x.begin()+view*width);
    for (int c = 0; c < 2; ++c) for (int p = 0; p < 6; ++p) {
      auto pieces = b.pieces(PieceType(static_cast<PieceType::underlying>(p)),(c == 0 ? Color::WHITE : Color::BLACK));
      while (pieces) add(a,p,c,pieces.pop(),1);
    }
    return a;
  }
  using Pieces = std::array<Bitboard,12>;
  static Pieces pieces(const Board &b) {
    Pieces result;
    for (int c = 0; c < 2; ++c) for (int p = 0; p < 6; ++p)
      result[c*6+p] = b.pieces(PieceType(static_cast<PieceType::underlying>(p)),(c == 0 ? Color::WHITE : Color::BLACK));
    return result;
  }
  void update(Accumulator &a, const Pieces &before, const Board &after) const {
    auto now = pieces(after);
    for (int c = 0; c < 2; ++c) for (int p = 0; p < 6; ++p) {
      int i = c*6+p;
      auto removed = before[i] & ~now[i], added = now[i] & ~before[i];
      while (removed) add(a,p,c,removed.pop(),-1);
      while (added) add(a,p,c,added.pop(),1);
    }
  }
  float evaluate(const Board &b, int ep, const Accumulator &a) const {
    if(evaluation_mode==2)return 0;
    if(evaluation_mode==1)return std::clamp(std::tanh(b.sideToMove()==Color::BLACK?-a.material:a.material),-.9999f,.9999f);
    std::array<float,2060> x;
    int side = b.sideToMove() == Color::BLACK;
    for (int j = 0; j < width; ++j) {
      x[j] = std::clamp(a.x[side*width+j],0.f,1.f);
      x[width+j] = std::clamp(a.x[(1-side)*width+j],0.f,1.f);
    }
    // Match CheapModel's effective metadata; avoid encoding 64 tokens and
    // history scans on every cheap evaluation.
    using S = Board::CastlingRights::Side;
    auto cr = b.castlingRights(); auto stm = b.sideToMove();
    float *m = x.data()+2*width;
    std::fill_n(m,12,0.f);
    m[0]=cr.has(stm,S::KING_SIDE); m[1]=cr.has(stm,S::QUEEN_SIDE);
    m[2]=cr.has(~stm,S::KING_SIDE); m[3]=cr.has(~stm,S::QUEEN_SIDE);
    m[4]=ep < 0 ? -1.f : float(ep ^ (side ? 56 : 0))/63.f;
    m[8]=b.inCheck(); m[9]=!side; m[11]=ep >= 0;
    int dim = 2*width+12;
    float result = output_b + (side ? -a.material : a.material);
    for (int i = 0; i < hidden; ++i) {
      const float *w = hidden_w.data()+i*dim;
      float h=hidden_b[i]+dot(w,x.data(),dim);
      result += output[i]*std::max(0.f,h);
    }
    return std::clamp(std::tanh(result),-.9999f,.9999f);
  }
};
struct Node {
  int parent = -1, depth = 0, ep = -1, solved = Unknown, visits = 0;
  Move move;
  float value = 0, cheap = 0, prior = 0, anchor = 0, anchor_weight = 0;
  bool repeated = false, expanded = false, frozen = false, corrected = false;
  std::vector<Move> moves;
  std::vector<int> children;
  std::vector<float> priors, neural_q;
  float reference=0, applied_correction=0, cpu_value=0;
  int cpu_depth=0;
  bool anchor_clipped=false, cpu_ready=false;
  float q_prior=NAN;
  double neural_sum=0;
  int neural_count=0;
  bool evidence=false;
};
// Independent bounded CPU searches. Bounded attacker search can PROVE a mate,
// but failure/timeout never disproves it. Every defending move is tested.
// No selective value estimate below is allowed to create a proof.
struct Tactical {
  const Weights &w;
  int mate_nodes=0, value_nodes=0, evaluations=0, mate_limit=0, value_limit=0;
  struct Ordered { Move move; int score; bool check; };
  explicit Tactical(const Weights &weights):w(weights){}
  static std::vector<Ordered> order(Board &b,const Movelist &ms) {
    static constexpr int v[]={100,320,330,500,900,20000,0};
    std::vector<Ordered> result;
    for(auto m:ms) {
      int captured=m.typeOf()==Move::ENPASSANT?100:v[int(b.at(m.to()).type())];
      int score=16*captured-v[int(b.at(m.from()).type())]+(m.typeOf()==Move::PROMOTION?10000:0);
      bool check=b.givesCheck(m)!=CheckType::NO_CHECK;
      result.push_back({m,score+(check?100000:0),check});
    }
    std::stable_sort(result.begin(),result.end(),[](auto a,auto b){return a.score>b.score;});
    return result;
  }
  bool mate(Board &b,Color attacker,int depth,bool repeated) {
    if(mate_nodes>=mate_limit)return false;
    ++mate_nodes;
    Movelist ms;movegen::legalmoves(ms,b);
    int terminal=nc_rules::terminal(b,ms,repeated);
    if(terminal!=Unknown)return terminal==-1 && b.sideToMove()!=attacker;
    if(depth<=0)return false;
    bool attacking=b.sideToMove()==attacker;
    auto moves=order(b,ms);
    for(auto x:moves) {
      // A mate in one must give check. At greater depths include quiet threats
      // as well: the motivating Qxf7 move is a capture, not a check.
      if(attacking && depth==1 && !x.check)continue;
      b.makeMove(x.move);
      bool proven=mate(b,attacker,depth-1,repeated_after(b,repeated));
      b.unmakeMove(x.move);
      if(attacking && proven)return true;
      if(!attacking && !proven)return false;
      if(mate_nodes>=mate_limit)return false;
    }
    return !attacking;
  }
  struct Value { float value; bool complete; };
  Value search(Board &b,const Accumulator &acc,int ep,bool repeated,int depth,int qdepth,
               int extensions,float alpha,float beta,int ply=0) {
    if(value_nodes>=value_limit)return {0,false};
    ++value_nodes;
    Movelist ms;movegen::legalmoves(ms,b);
    int terminal=nc_rules::terminal(b,ms,repeated);
    if(terminal!=Unknown)return {float(terminal),true};
    bool check=b.inCheck();
    if(depth<=0 && (check || ms.size()==1) && extensions>0){depth=1;--extensions;}
    float stand=w.evaluate(b,ep,acc);++evaluations;
    if(ply>=12)return {stand,false};
    if(depth<=0 && qdepth<=0)return {stand,!check};
    bool quiet=depth<=0;
    float best=quiet && !check?stand:-2.f;
    if(quiet && !check){if(stand>=beta)return {stand,true};alpha=std::max(alpha,stand);}
    auto moves=order(b,ms);
    for(auto x:moves) {
      if(quiet && !check && !b.isCapture(x.move) && x.move.typeOf()!=Move::PROMOTION && !(qdepth==2 && x.check))continue;
      auto before=Weights::pieces(b);Accumulator next_acc;w.copy(next_acc,acc);int next_ep=nc_rules::nextEp(b,x.move);
      b.makeMove(x.move);w.update(next_acc,before,b);
      auto result=search(b,next_acc,next_ep,repeated_after(b,repeated),depth-1,
                         quiet?qdepth-1:qdepth,extensions,-beta,-alpha,ply+1);
      b.unmakeMove(x.move);
      if(!result.complete)return {0,false};
      float value=-result.value;best=std::max(best,value);alpha=std::max(alpha,value);
      if(alpha>=beta)break;
    }
    return {best,true};
  }
};
struct Engine {
  Weights weights;
  Config c;
  Board root;
  Accumulator root_acc;
  std::vector<Node> nodes;
  std::vector<int> pending;
  int expansions = 0, cheap_evals = 0, queries = 0, rounds = 0, next_round = 1;
  int terminals = 0, max_depth = 0;
  int qnodes = 0;
  bool finished = false, node_limit = false;
  std::string result;
  struct Correction { int id, expansion; float before, neural, after; };
  std::vector<Correction> corrections;
  float direct_weight=.75f, q_weight=.25f, decay=0.f, root_policy_weight=0.f;
  int scheduler=0; bool align_root=false, verify_final=false, local_exploration=false;
  int verification_reserve=8, verification_queries=0;
  float correction_clip=0, root_value_clip=0, cpu_correction_clip=0;
  int mate_budget=0,mate_depth=7,tactical_budget=0,tactical_depth=2;
  int mate_nodes=0,tactical_nodes=0,cpu_proofs=0,cpu_references=0;
  int backup_mode=0,tree_reuses=0,retained_nodes=0;
  int root_coverage=0;
  bool reuse_tree=false;
  std::string initial;
  std::vector<std::string> root_history;
  // Only neural observations (or supported parent Q) form calibrated evidence.
  // Cheap evaluations still guide expansion when evidence is unavailable.
  void evidence_value(Node &n) const {
    n.evidence=n.corrected || std::isfinite(n.q_prior) || n.neural_count>0 || n.solved!=Unknown;
    if(n.solved!=Unknown){n.value=float(n.solved);return;}
    if(backup_mode==3 && n.neural_count) {
      double prior=std::isfinite(n.q_prior)?2.:0.;
      n.value=float((n.neural_sum+(prior?n.q_prior*prior:0.))/(n.neural_count+prior));
    } else if(n.corrected)n.value=n.anchor;
    else if(std::isfinite(n.q_prior))n.value=n.q_prior;
    else n.value=n.cpu_ready?n.cpu_value:n.cheap;
  }
  float fuse(Node &n,float base) const {
    n.reference=base;
    float delta=n.anchor-base;
    float clip=correction_clip>0?correction_clip:n.cpu_ready?cpu_correction_clip:0;
    n.anchor_clipped=clip>0 && std::abs(delta)>clip && n.anchor_weight>0;
    if(clip>0)delta=std::clamp(delta,-clip,clip);
    n.applied_correction=anchor_weight(n)*delta;
    return clip>0?base+n.applied_correction:(1-anchor_weight(n))*base+anchor_weight(n)*n.anchor;
  }
  void leaf_value(Node &n,bool direct=false) const {
    if(backup_mode>=2){evidence_value(n);return;}
    float base=n.cpu_ready?n.cpu_value:n.cheap;
    n.value=direct && correction_clip==0 && !n.cpu_ready?n.anchor:fuse(n,base);
  }
  float decision_value(int child) const {
    const auto &n=nodes[child];
    float value=-n.value,reference=-n.cheap;
    if(root_value_clip>0 && n.solved==Unknown)
      value=std::clamp(value,reference-root_value_clip,reference+root_value_clip);
    return value;
  }
  float root_score(int child) const {
    return decision_value(child)+root_policy_weight*std::log(std::max(1e-8f,nodes[child].prior));
  }
  float anchor_weight(const Node &n) const {
    return decay>0 ? n.anchor_weight/(1.f+n.visits/decay) : n.anchor_weight;
  }
  int exploration_limit() const {
    return verify_final && c.queries>0 ? c.queries-std::min(verification_reserve,c.queries-1) : c.queries;
  }
  std::vector<int> rank_root(bool require_checked=false) const {
    auto rank=nodes[0].children;
    if(backup_mode>=2) {
      std::vector<int> evidence;
      for(int i:rank)if(nodes[i].evidence || nodes[i].solved!=Unknown)evidence.push_back(i);
      if(!evidence.empty())rank=std::move(evidence);
    }
    if(require_checked && c.queries>0) {
      std::vector<int> checked;
      for(int i:rank) if(nodes[i].solved!=1 && (nodes[i].corrected || nodes[i].solved!=Unknown)) checked.push_back(i);
      if(!checked.empty()) rank=std::move(checked);
      else {
        // No child evaluation was affordable: use the root policy, retaining
        // exact-outcome priority, instead of an unchecked optimistic value.
        std::vector<int> unresolved;
        for(int i:rank) if(nodes[i].solved==Unknown) unresolved.push_back(i);
        if(!unresolved.empty()) rank=std::move(unresolved);
        std::stable_sort(rank.begin(),rank.end(),[&](int a,int b){return nodes[a].prior>nodes[b].prior;});
        return rank;
      }
    }
    auto tier=[&](int i){return nodes[i].solved==-1 ? 2 : nodes[i].solved==1 ? 0 : 1;};
    std::stable_sort(rank.begin(),rank.end(),[&](int a,int b){
      if(tier(a)!=tier(b)) return tier(a)>tier(b);
      if(root_score(a)!=root_score(b)) return root_score(a)>root_score(b);
      if(nodes[a].visits!=nodes[b].visits) return nodes[a].visits>nodes[b].visits;
      return nodes[a].prior>nodes[b].prior;
    });
    return rank;
  }
  Engine(const char *path, Config config) : weights(path), c(config) {
    if (c.expansions < 1 || c.max_nodes < 257 || c.depth < 1 || c.depth > 256 ||
        c.queries < 0 || c.batch < 1 || c.batch > 256 || c.qdepth<0 || c.qdepth>8 || c.qbudget<0)
      throw std::runtime_error("Invalid best-first configuration");
  }
  float quiet(Board &b, const Accumulator &acc, int ep, bool repeated, int depth,
              float alpha, float beta, const Movelist *known=nullptr) {
    Movelist generated;
    if (!known) {
      movegen::legalmoves(generated,b); known=&generated;
      int solved=nc_rules::terminal(b,*known,repeated);
      if(solved!=Unknown) return float(solved);
    }
    float stand=weights.evaluate(b,ep,acc); ++cheap_evals;
    if (!depth || qnodes>=c.qbudget) return stand;
    bool check=b.inCheck();
    if (!check) { if(stand>=beta) return stand; alpha=std::max(alpha,stand); }
    std::vector<Move> moves;
    for(auto m:*known) if(check || b.isCapture(m) || m.typeOf()==Move::PROMOTION) moves.push_back(m);
    auto order=[&](Move m) {
      static constexpr int v[]={100,320,330,500,900,20000,0};
      int victim=m.typeOf()==Move::ENPASSANT ? 100 : v[int(b.at(m.to()).type())];
      return 10*victim-v[int(b.at(m.from()).type())]+(m.typeOf()==Move::PROMOTION ? 9000 : 0);
    };
    std::stable_sort(moves.begin(),moves.end(),[&](Move a,Move z){return order(a)>order(z);});
    float best=check ? -1.f : stand; bool searched=false;
    for(auto m:moves) {
      if(qnodes>=c.qbudget) break;
      auto before=Weights::pieces(b); Accumulator child;weights.copy(child,acc);int next_ep=nc_rules::nextEp(b,m);
      b.makeMove(m); weights.update(child,before,b); ++qnodes;
      float v=-quiet(b,child,next_ep,repeated_after(b,repeated),depth-1,-beta,-alpha);
      b.unmakeMove(m); searched=true; best=std::max(best,v); alpha=std::max(alpha,v);
      if(alpha>=beta) break;
    }
    return check && !searched ? stand : best;
  }
  int initialize(Board &b, const Accumulator &acc, int parent, Move move, int ep, bool repeated) {
    Node n; n.parent=parent; n.move=move; n.ep=ep; n.repeated=repeated;
    n.depth=parent < 0 ? 0 : nodes[parent].depth+1;
    Movelist ms; movegen::legalmoves(ms,b);
    n.solved=nc_rules::terminal(b,ms,repeated);
    n.moves.assign(ms.begin(),ms.end());
    if (n.solved != Unknown) { n.value=n.solved; n.frozen=true; ++terminals; }
    else { n.cheap=n.value=std::clamp(quiet(b,acc,ep,repeated,c.qdepth,-1,1,&ms),-.9999f,.9999f); }
    max_depth=std::max(max_depth,n.depth);
    nodes.push_back(std::move(n));
    return nodes.size()-1;
  }
  void start(const char *fen, const char *history) {
    if(!pending.empty())throw std::runtime_error("Pending corrections at start");
    std::istringstream history_in(history);std::string token;std::vector<std::string> hist;
    while(history_in>>token)hist.push_back(token);
    bool reused=reuse_tree && !nodes.empty() && initial==fen && hist.size()>=root_history.size() &&
                std::equal(root_history.begin(),root_history.end(),hist.begin());
    int adopt=0;
    if(reused)for(size_t k=root_history.size();k<hist.size();++k) {
      int child=-1;
      for(int j:nodes[adopt].children)if(uci::moveToUci(nodes[j].move)==hist[k]){child=j;break;}
      if(child<0){reused=false;break;}adopt=child;
    }
    retained_nodes=tree_reuses=0;
    if(reused) {
      std::vector<int> ids{adopt};
      for(size_t i=0;i<ids.size();++i)for(int j:nodes[ids[i]].children)ids.push_back(j);
      // Leave ample room for the unchanged per-move expansion budget.
      if(ids.size()>size_t(c.max_nodes/4))reused=false;
      else {
        std::vector<int> mapping(nodes.size(),-1);
        for(size_t i=0;i<ids.size();++i)mapping[ids[i]]=i;
        int depth=nodes[adopt].depth;std::vector<Node> kept;kept.reserve(c.max_nodes);
        for(int i:ids) {
          Node n=std::move(nodes[i]);n.depth-=depth;n.parent=i==adopt?-1:mapping[n.parent];
          for(auto &j:n.children)j=mapping[j];
          // An old depth cutoff is not a proof and has a new horizon now.
          n.frozen=n.solved!=Unknown;kept.push_back(std::move(n));
        }
        nodes=std::move(kept);retained_nodes=nodes.size();tree_reuses=1;
      }
    }
    if(!reused)nodes.clear();
    initial=fen;root_history=hist;
    pending.clear(); corrections.clear(); expansions=cheap_evals=queries=rounds=terminals=max_depth=qnodes=0;
    finished=node_limit=false; next_round=1;verification_queries=0;
    mate_nodes=tactical_nodes=cpu_proofs=cpu_references=0;
    root=Board(fen); int ep=nc_rules::rawEp(fen); bool repeated=false;
    std::istringstream in(history); std::string text;
    while (in >> text) {
      Move m=uci::uciToMove(root,text); Movelist legal; movegen::legalmoves(legal,root);
      if (std::find(legal.begin(),legal.end(),m)==legal.end())
        throw std::runtime_error("Illegal root history");
      ep=nc_rules::nextEp(root,m); root.makeMove(m); repeated=repeated_after(root,repeated);
    }
    root_acc=weights.initial(root);
    nodes.reserve(c.max_nodes);
    if(!reused)initialize(root,root_acc,-1,Move(),ep,repeated);
    else {
      for(int i=int(nodes.size())-1;i>=0;--i)if(nodes[i].expanded)backup(i);
      for(const auto &n:nodes)max_depth=std::max(max_depth,n.depth);
    }
    if(!reused) {
      if(nodes[0].solved!=Unknown)throw std::runtime_error("Terminal root");
    } else {
      // A retained proof concerns the future game outcome, not necessarily an
      // already terminal board. It must remain playable without another NN.
      Movelist legal;movegen::legalmoves(legal,root);
      if(nc_rules::terminal(root,legal,repeated)!=Unknown)throw std::runtime_error("Terminal root");
    }
    int limit=exploration_limit();
    rounds = limit > 1 ? (limit-1+c.batch-1)/c.batch : 0;
    if(mate_budget || tactical_budget) {
      if(!nodes[0].expanded)expand_one();
      tactical_root();
    }
  }
  void tactical_root() {
    Tactical cpu(weights);
    int count=nodes[0].children.size();
    if(!count)return;
    for(int child:nodes[0].children) {
      auto &n=nodes[child];if(n.solved!=Unknown)continue;
      Board b=root;Accumulator acc;weights.copy(acc,root_acc);auto before=Weights::pieces(b);
      b.makeMove(n.move);weights.update(acc,before,b);
      if(mate_budget) {
        int quota=mate_budget/(2*count);
        for(int direction=0;direction<2 && n.solved==Unknown;++direction) {
          int allotted=std::min(mate_budget,cpu.mate_nodes+quota);cpu.mate_limit=allotted;
          Color attacker=direction?root.sideToMove():b.sideToMove();
          for(int d=1;d<=mate_depth;d+=2) {
            if(cpu.mate(b,attacker,d-(direction?1:0),n.repeated)) {
              n.solved=direction?-1:1;n.value=n.solved;n.frozen=true;++cpu_proofs;break;
            }
            if(cpu.mate_nodes>=allotted)break;
          }
        }
      }
      if(n.solved==Unknown && tactical_budget) {
        int allotted=std::min(tactical_budget,cpu.value_nodes+tactical_budget/count);
        cpu.value_limit=allotted;
        for(int d=1;d<=tactical_depth;++d) {
          auto result=cpu.search(b,acc,n.ep,n.repeated,d,2,2,-1,1);
          if(!result.complete)break;
          n.cpu_value=std::clamp(result.value,-.9999f,.9999f);n.cpu_ready=true;n.cpu_depth=d;
        }
        if(n.cpu_ready) {
          // A retained child can already have a searched frontier. Refresh its
          // CPU reference without replacing the backed subtree with a leaf.
          if(n.expanded)backup(child);else leaf_value(n);
          ++cpu_references;
        }
      }
      if(n.solved==-1)break;
    }
    mate_nodes=cpu.mate_nodes;tactical_nodes=cpu.value_nodes;cheap_evals+=cpu.evaluations;backup(0);
  }
  void backup(int id) {
    while (id >= 0) {
      auto &n=nodes[id];
      if (n.expanded) {
        bool all=true, blocked=true; int proof=-1; float best=-2, trusted=-2;
        double mean_sum=0,mean_weight=0;
        for (int child : n.children) {
          const auto &x=nodes[child]; best=std::max(best,-x.value);
          if(x.evidence || x.solved!=Unknown)trusted=std::max(trusted,-x.value);
          if(x.visits){mean_sum-=double(x.value)*x.visits;mean_weight+=x.visits;}
          all=all && x.solved != Unknown; blocked=blocked && x.frozen;
          if (x.solved != Unknown) proof=std::max(proof,-x.solved);
          if (x.solved == -1) { n.solved=1; break; }
        }
        if (n.solved == Unknown && all) n.solved=proof;
        // A large-model correction is an evaluation of this position, not a
        // disposable visit. Retain it when cheap descendants are expanded.
        // This is explicitly regularized minimax, not an exact score bound.
        float base=n.cpu_ready?.5f*best+.5f*n.cpu_value:best;
        if(backup_mode==1 && mean_weight)base=float(mean_sum/mean_weight);
        if(backup_mode>=2) {
          evidence_value(n);
          if(backup_mode==2 && trusted>=-1.f && n.solved==Unknown) {
            n.value=n.corrected?(1-direct_weight)*trusted+direct_weight*n.anchor:trusted;
            n.evidence=true;
          }
        } else n.value=n.solved == Unknown ? fuse(n,base) : float(n.solved);
        n.frozen=blocked || n.solved != Unknown;
      }
      id=n.parent;
    }
  }
  bool expand_one() {
    if (nodes[0].frozen) return false;
    Board b=root; Accumulator acc;weights.copy(acc,root_acc);
    int id=0; std::vector<int> path{0};
    while (nodes[id].expanded) {
      int chosen=-1; float best=-1e30f;
      // A global clock can route every exploration to the root's other branch,
      // so a heavily visited principal branch never explores its own siblings.
      bool explore=((local_exploration ? nodes[id].visits : expansions)%16 == 15);
      for (int child : nodes[id].children) {
        const auto &x=nodes[child]; if (x.frozen) continue;
        float score = explore ? -float(x.visits) + (local_exploration ? .01f*x.prior+.001f*(-x.value) : .01f*(-x.value)) :
                      -x.value + .015f*x.prior/(1+x.visits);
        if(!explore && id==0 && align_root) score=root_score(child);
        if (score > best) { best=score; chosen=child; }
      }
      if (chosen < 0) { nodes[id].frozen=true; backup(nodes[id].parent); return true; }
      auto before=Weights::pieces(b); b.makeMove(nodes[chosen].move);
      weights.update(acc,before,b); id=chosen; path.push_back(id);
    }
    if (nodes[id].depth >= c.depth) {
      nodes[id].frozen=true; backup(nodes[id].parent); return true;
    }
    auto moves=nodes[id].moves;
    if (nodes.size()+moves.size() > size_t(c.max_nodes)) { node_limit=true; return false; }
    // Every legal child is materialized. Approximate scores never prove a loss.
    for (size_t j=0; j<moves.size(); ++j) {
      Move m=moves[j]; int ep=nc_rules::nextEp(b,m);
      auto before=Weights::pieces(b);Accumulator child_acc;weights.copy(child_acc,acc);
      b.makeMove(m); weights.update(child_acc,before,b);
      bool repeated=repeated_after(b,nodes[id].repeated);
      int child=initialize(b,child_acc,id,m,ep,repeated);
      b.unmakeMove(m);
      nodes[id].children.push_back(child);
      nodes[child].prior=nodes[id].priors.empty() ? 1.f/moves.size() : nodes[id].priors[j];
      if (!nodes[id].neural_q.empty() && std::isfinite(nodes[id].neural_q[j]) && nodes[child].solved==Unknown)
      {
        nodes[child].anchor=-nodes[id].neural_q[j];
        nodes[child].q_prior=nodes[child].anchor;
        nodes[child].anchor_weight=q_weight;
        leaf_value(nodes[child]);
      }
    }
    nodes[id].expanded=true;
    for (int n : path) ++nodes[n].visits;
    ++expansions; backup(id); return true;
  }
  struct Candidate { int id; float priority; };
  void candidates(int id, float regret, std::vector<Candidate> &out) const {
    if (out.size() >= 4096 || nodes[id].solved != Unknown) return;
    const auto &n=nodes[id];
    // Calibrate high-level candidate positions before following their cheap
    // principal variations arbitrarily deep. Expansion is not evidence that
    // a position's evaluation has been checked by the large model.
    if (!n.corrected && (scheduler==0 || !n.expanded)) {
      out.push_back({id,-regret-.001f*n.depth});
      return;
    }
    if (!n.expanded) return;
    std::vector<int> rank=n.children;
    std::stable_sort(rank.begin(),rank.end(),[&](int a,int b){
      return id==0 && align_root ? root_score(a)>root_score(b) : nodes[a].value<nodes[b].value;
    });
    // Counterfactual relevance: cumulative local minimax gaps. This is a
    // heuristic scheduler, not a confidence interval or a proof bound.
    for (int child : rank) {
      float gap=id==0 && align_root ? std::max(0.f,root_score(rank[0])-root_score(child)) :
                                     std::max(0.f,n.value+nodes[child].value);
      candidates(child,regret+gap,out);
      if (out.size() >= 4096) break;
    }
  }
  int emit(const std::vector<int> &selected,int cap,uint8_t *tokens,float *meta,int16_t *ids) {
    int count=std::min({cap,c.batch,c.queries-queries,int(selected.size())});
    for (int i=0; i<count; ++i) {
      int id=selected[i]; Board b=root; std::vector<int> path;
      for (int x=id; nodes[x].parent>=0; x=nodes[x].parent) path.push_back(x);
      for (auto it=path.rbegin(); it!=path.rend(); ++it) b.makeMove(nodes[*it].move);
      nc_rules::encode(b,nodes[id].ep,tokens+i*64,meta+i*12);
      std::fill(ids+i*Capacity,ids+(i+1)*Capacity,int16_t(-1));
      if (nodes[id].moves.size()>Capacity) throw std::runtime_error("Legal capacity exceeded");
      for (size_t j=0;j<nodes[id].moves.size();++j)
        ids[i*Capacity+j]=nc_rules::action(nodes[id].moves[j],b.sideToMove());
      pending.push_back(id);
    }
    return count;
  }
  int query(int cap,uint8_t *tokens,float *meta,int16_t *ids) {
    std::vector<Candidate> list;
    if(!queries) list.push_back({0,0});
    else {
      // Reserve early calibration for plausible root alternatives. A noisy
      // cheap/CPU value must not exclude a high-policy move before it gets
      // even one direct neural observation. These queries use the SAME cap.
      auto children=nodes[0].children;
      std::stable_sort(children.begin(),children.end(),[&](int a,int b){return nodes[a].prior>nodes[b].prior;});
      for(int j=0;j<std::min(root_coverage,int(children.size()));++j) {
        int id=children[j];
        if(!nodes[id].corrected && nodes[id].solved==Unknown)list.push_back({id,-float(j)});
      }
      if(list.empty())candidates(0,0,list);
    }
    std::stable_sort(list.begin(),list.end(),[](auto a,auto b){return a.priority>b.priority;});
    int count=std::min(int(list.size()),exploration_limit()-queries);
    std::vector<int> selected;
    for(int i=0;i<count;++i) selected.push_back(list[i].id);
    return emit(selected,cap,tokens,meta,ids);
  }
  int verify(int cap,uint8_t *tokens,float *meta,int16_t *ids) {
    if(!verify_final || nodes[0].solved!=Unknown || queries>=c.queries || nodes[0].children.empty()) return 0;
    auto rank=rank_root();int id=rank[0];
    if(nodes[id].corrected || nodes[id].solved!=Unknown) return 0;
    int count=emit({id},cap,tokens,meta,ids);verification_queries+=count;return count;
  }
  int next(int cap, uint8_t *tokens, float *meta, int16_t *ids) {
    if (!pending.empty()) throw std::runtime_error("Submit pending corrections first");
    if (finished) return 0;
    if (cap<1) throw std::runtime_error("Positive query capacity required");
    if(nodes[0].solved!=Unknown){finished=true;return 0;}
    if (c.queries && !queries) return query(cap,tokens,meta,ids);
    while (true) {
      int target=rounds ? (c.expansions*std::min(next_round,rounds)+rounds-1)/rounds : c.expansions;
      while (expansions<target && !nodes[0].frozen && !node_limit)
        if (!expand_one()) break;
      bool end=expansions>=c.expansions || nodes[0].frozen || node_limit;
      if (queries<exploration_limit() && nodes[0].solved==Unknown && rounds && next_round<=rounds) {
        ++next_round;
        int n=query(cap,tokens,meta,ids);
        if (n) return n;
      }
      if (end || next_round>rounds) {
        int n=verify(cap,tokens,meta,ids);if(n)return n;
        finished=true; return 0;
      }
    }
  }
  void submit(const float *policy, const float *q, const float *values, int count) {
    if (count != int(pending.size()) || count<1) throw std::runtime_error("Correction batch mismatch");
    for (int i=0;i<count;++i) {
      auto &n=nodes[pending[i]];
      float previous=n.value;
      if (!std::isfinite(values[i]) || std::abs(values[i])>1.0001f)
        throw std::runtime_error("Invalid correction value");
      float mx=-INFINITY, sum=0;
      for (size_t j=0;j<n.moves.size();++j) {
        float p=policy[i*Capacity+j];
        if (!std::isfinite(p)) throw std::runtime_error("Invalid correction policy");
        mx=std::max(mx,p);
      }
      n.priors.resize(n.moves.size()); n.neural_q.resize(n.moves.size(),NAN);
      for (size_t j=0;j<n.moves.size();++j) {
        n.priors[j]=std::exp(policy[i*Capacity+j]-mx); sum+=n.priors[j];
        if (q) {
          float v=q[i*Capacity+j];
          if (std::isinf(v) || (std::isfinite(v) && std::abs(v)>1.0001f))
            throw std::runtime_error("Invalid correction Q");
          n.neural_q[j]=v;
        }
      }
      for (auto &p:n.priors) p/=sum;
      if(!n.corrected) {
        float value=std::clamp(values[i],-.9999f,.9999f);
        for(int x=pending[i];x>=0;x=nodes[x].parent){nodes[x].neural_sum+=value;++nodes[x].neural_count;value=-value;}
      }
      n.corrected=true; n.anchor=std::clamp(values[i],-.9999f,.9999f);
      n.anchor_weight=direct_weight;leaf_value(n,true);
      // Reuse this query's action priors for already materialized children.
      for (size_t j=0;j<n.children.size();++j) {
        auto &child=nodes[n.children[j]]; child.prior=n.priors[j];
        if (!child.corrected && child.solved==Unknown && std::isfinite(n.neural_q[j])) {
          child.anchor=-n.neural_q[j]; child.anchor_weight=q_weight;
          child.q_prior=child.anchor;
          if (child.expanded) backup(n.children[j]);
          else leaf_value(child);
        }
      }
      ++queries; backup(pending[i]);
      corrections.push_back({pending[i],expansions,previous,values[i],n.value});
    }
    pending.clear();
  }
  const char *get_result() {
    if (!finished || nodes[0].children.empty()) throw std::runtime_error("Search not complete");
    auto rank=rank_root(verify_final);
    auto score=[&](int i) {return root_score(i);};
    int clipped=std::count_if(nodes.begin(),nodes.end(),[](const Node &n){return n.anchor_clipped;});
    std::ostringstream s;
    s << "{\"move\":\"" << uci::moveToUci(nodes[rank[0]].move) << "\",\"value\":" << nodes[0].value
      << ",\"selected_value\":" << -nodes[rank[0]].value << ",\"decision_value\":" << decision_value(rank[0])
      << ",\"decision_score\":" << score(rank[0])
      << ",\"final_candidate_checked\":" << ((nodes[rank[0]].corrected || nodes[rank[0]].solved!=Unknown)?"true":"false")
      << ",\"proven_value\":" << (nodes[0].solved==Unknown ? "null" : std::to_string(nodes[0].solved))
      << ",\"stats\":{\"expanded_nodes\":" << expansions << ",\"cheap_evaluations\":" << cheap_evals
      << ",\"correction_positions\":" << queries << ",\"nodes\":" << nodes.size()
      << ",\"terminal_nodes\":" << terminals << ",\"max_depth\":" << max_depth
      << ",\"quiescence_nodes\":" << qnodes
      << ",\"verification_queries\":" << verification_queries
      << ",\"mate_nodes\":" << mate_nodes << ",\"tactical_nodes\":" << tactical_nodes
      << ",\"cpu_mate_proofs\":" << cpu_proofs << ",\"cpu_references\":" << cpu_references
      << ",\"clipped_anchor_nodes\":" << clipped
      << ",\"tree_reuses\":" << tree_reuses << ",\"retained_nodes\":" << retained_nodes
      << ",\"node_limit_hit\":" << (node_limit ? "true" : "false") << "},\"leaders\":[";
    for (int k=0;k<std::min(8,int(rank.size()));++k) {
      if(k) s << ',';
      auto &n=nodes[rank[k]];
      s << "{\"move\":\"" << uci::moveToUci(n.move) << "\",\"value\":" << -n.value
        << ",\"decision_score\":" << score(rank[k]) << ",\"prior\":" << n.prior
        << ",\"expansions\":" << n.visits << ",\"corrected\":" << (n.corrected ? "true" : "false") << '}';
    }
    s << "]}"; result=s.str(); return result.c_str();
  }
  const char *trace() {
    std::ostringstream s; s.precision(9); s << "{\"nodes\":[";
    for(size_t i=0;i<nodes.size();++i) {
      const auto &n=nodes[i]; if(i) s << ',';
      s << "{\"id\":" << i << ",\"parent\":" << n.parent << ",\"move\":\""
        << (n.parent<0 ? "" : uci::moveToUci(n.move)) << "\",\"depth\":" << n.depth
        << ",\"value\":" << n.value << ",\"cheap\":" << n.cheap << ",\"anchor\":" << n.anchor
        << ",\"anchor_weight\":" << n.anchor_weight << ",\"prior\":" << n.prior
        << ",\"q_prior\":" << (std::isfinite(n.q_prior)?std::to_string(n.q_prior):"null")
        << ",\"neural_sum\":" << n.neural_sum << ",\"neural_count\":" << n.neural_count
        << ",\"evidence\":" << (n.evidence?"true":"false")
        << ",\"effective_anchor_weight\":" << anchor_weight(n)
        << ",\"visits\":" << n.visits << ",\"solved\":" << n.solved
        << ",\"expanded\":" << (n.expanded?"true":"false")
        << ",\"corrected\":" << (n.corrected?"true":"false")
        << ",\"reference\":" << n.reference << ",\"applied_correction\":" << n.applied_correction
        << ",\"anchor_clipped\":" << (n.anchor_clipped?"true":"false")
        << ",\"cpu_value\":" << n.cpu_value << ",\"cpu_depth\":" << n.cpu_depth
        << ",\"cpu_ready\":" << (n.cpu_ready?"true":"false") << '}';
    }
    s << "],\"corrections\":[";
    for(size_t i=0;i<corrections.size();++i) {
      auto e=corrections[i]; if(i) s << ',';
      s << "{\"id\":" << e.id << ",\"expansion\":" << e.expansion << ",\"before\":" << e.before
        << ",\"neural\":" << e.neural << ",\"after\":" << e.after << '}';
    }
    s << "]}"; result=s.str(); return result.c_str();
  }
};
thread_local std::string error;
template<class F> int protect(F f) {
  try { f(); error.clear(); return 0; }
  catch(const std::exception &e) { error=e.what(); return -1; }
}
} // namespace bf
extern "C" {
const char *bf_error() { return bf::error.c_str(); }
void *bf_create(const char *weights, bf::Config c) {
  try { return new bf::Engine(weights,c); }
  catch(const std::exception &e) { bf::error=e.what(); return nullptr; }
}
void bf_destroy(void *p) { delete static_cast<bf::Engine *>(p); }
int bf_ablation(void *p,int reuse,int backup,int evaluator) {
  return bf::protect([&]{
    if(reuse<0 || reuse>1 || backup<0 || backup>3 || evaluator<0 || evaluator>2)
      throw std::runtime_error("Invalid ablation options");
    auto &e=*static_cast<bf::Engine *>(p);
    e.reuse_tree=reuse;e.backup_mode=backup;e.weights.evaluation_mode=evaluator;
  });
}
int bf_root_coverage(void *p,int count) {
  return bf::protect([&]{
    if(count<0 || count>256)throw std::runtime_error("Invalid root coverage");
    static_cast<bf::Engine *>(p)->root_coverage=count;
  });
}
int bf_forget(void *p) {
  return bf::protect([&]{
    auto &e=*static_cast<bf::Engine *>(p);
    if(!e.pending.empty())throw std::runtime_error("Pending corrections at forget");
    e.nodes.clear();e.initial.clear();e.root_history.clear();
  });
}
int bf_local_exploration(void *p,int enabled) {
  return bf::protect([&]{
    if(enabled<0 || enabled>1) throw std::runtime_error("Invalid local exploration option");
    static_cast<bf::Engine *>(p)->local_exploration=enabled;
  });
}
int bf_safety(void *p,float clip,int mate_budget,int mate_depth,int tactical_budget,int tactical_depth) {
  return bf::protect([&]{
    if(!std::isfinite(clip) || clip<0 || clip>1 || mate_budget<0 || mate_budget>1000000 ||
       mate_depth<1 || mate_depth>9 || tactical_budget<0 || tactical_budget>1000000 ||
       tactical_depth<1 || tactical_depth>3)throw std::runtime_error("Invalid safety options");
    auto &e=*static_cast<bf::Engine *>(p);
    e.correction_clip=clip;e.mate_budget=mate_budget;e.mate_depth=mate_depth;
    e.tactical_budget=tactical_budget;e.tactical_depth=tactical_depth;
  });
}
int bf_root_clip(void *p,float clip) {
  return bf::protect([&]{
    if(!std::isfinite(clip) || clip<0 || clip>1)throw std::runtime_error("Invalid root safety clip");
    static_cast<bf::Engine *>(p)->root_value_clip=clip;
  });
}
int bf_cpu_clip(void *p,float clip) {
  return bf::protect([&]{
    if(!std::isfinite(clip) || clip<0 || clip>1)throw std::runtime_error("Invalid CPU safety clip");
    static_cast<bf::Engine *>(p)->cpu_correction_clip=clip;
  });
}
int bf_start(void *p,const char *fen,const char *moves) {
  return bf::protect([&]{static_cast<bf::Engine *>(p)->start(fen,moves);});
}
int bf_options(void *p,float direct,float q,float decay,int scheduler,float root_policy,int align_root,int verify,int reserve) {
  return bf::protect([&]{
    if(!std::isfinite(direct) || direct<0 || direct>1 || !std::isfinite(q) || q<0 || q>1 ||
       !std::isfinite(decay) || decay<0 || scheduler<0 || scheduler>1 ||
       !std::isfinite(root_policy) || root_policy<0 || align_root<0 || align_root>1 ||
       verify<0 || verify>1 || reserve<0 || reserve>256)
      throw std::runtime_error("Invalid search options");
    auto &e=*static_cast<bf::Engine *>(p);
    e.direct_weight=direct; e.q_weight=q; e.decay=decay; e.scheduler=scheduler;e.root_policy_weight=root_policy;
    e.align_root=align_root;
    e.verify_final=verify;e.verification_reserve=reserve;
  });
}
int bf_next(void *p,int cap,uint8_t *t,float *m,int16_t *ids) {
  int n=0; int rc=bf::protect([&]{n=static_cast<bf::Engine *>(p)->next(cap,t,m,ids);});
  return rc<0 ? -1 : n;
}
int bf_submit(void *p,const float *policy,const float *q,const float *values,int n) {
  return bf::protect([&]{static_cast<bf::Engine *>(p)->submit(policy,q,values,n);});
}
const char *bf_result(void *p) {
  const char *s=nullptr; bf::protect([&]{s=static_cast<bf::Engine *>(p)->get_result();}); return s;
}
const char *bf_trace(void *p) {
  const char *s=nullptr; bf::protect([&]{s=static_cast<bf::Engine *>(p)->trace();}); return s;
}
// Exercise incremental updates and compare against complete accumulator rebuilds.
int bf_probe(void *p,const char *fen,const char *history,float *value,float *max_error) {
  return bf::protect([&]{
    auto &w=static_cast<bf::Engine *>(p)->weights; chess::Board b(fen);
    auto a=w.initial(b); int ep=nc_rules::rawEp(fen); *max_error=0;
    std::istringstream in(history); std::string text;
    while(in >> text) {
      auto m=chess::uci::uciToMove(b,text); auto before=bf::Weights::pieces(b);
      ep=nc_rules::nextEp(b,m); b.makeMove(m); w.update(a,before,b);
      auto full=w.initial(b);
      for(int i=0;i<2*w.width;++i) *max_error=std::max(*max_error,std::abs(a.x[i]-full.x[i]));
      *max_error=std::max(*max_error,std::abs(a.material-full.material));
    }
    *value=w.evaluate(b,ep,a);
  });
}
}
