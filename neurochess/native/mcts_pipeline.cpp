// Experimental out-of-order input-cache pipeline; baseline source remains frozen.
// Native PUCT core. No Python objects/callbacks in traversal, movegen or
// backup.
#include "chess_rules.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
using namespace chess;
using nc_rules::rawEp;
using nc_rules::nextEp;
using nc_rules::action;
using nc_rules::insufficient;
using nc_rules::terminal;
using nc_rules::encode;
namespace {
constexpr int Unknown = 2, Capacity = 256;
struct Config {
  int simulations, batch, width, keep, depth;
  float cpuct, fpu, widening, prune;
};
struct Edge {
  Move move;
  float prior = 0, total = 0, neural_q = NAN;
  int visits = 0, child = -1;
  bool forcing = false, pruned = false;
  float q() const { return visits ? total / visits : 0; }
};
struct Node {
  int begin = 0, count = 0, visits = 0, solved = Unknown;
  float value = 0;
  bool expanded = false, pending = false, pruned = false;
};
struct Stats {
  uint64_t choices = 0, evals = 0, sims = 0, terminals = 0, pruned = 0,
           skips = 0, mates = 0, depth = 0, limits = 0, reuse = 0, retained = 0,
           batches = 0;
};
struct Tree {
  Board board, work;
  int ep = -1;
  bool repeated = false;
  std::string initial;
  std::vector<std::string> history;
  std::vector<Node> nodes;
  std::vector<Edge> edges;
  int done = 0, inflight = 0, base = 0;
  explicit Tree(const std::string &fen)
      : board(fen), work(fen), ep(rawEp(fen)), initial(fen) {
    nodes.reserve(4096);
    edges.reserve(131072);
    nodes.emplace_back();
  }
  int node() {
    nodes.emplace_back();
    return int(nodes.size() - 1);
  }
  // Compact the retained subtree, releasing siblings and old roots after each
  // move.
  void adopt(int root) {
    std::vector<Node> ns;
    std::vector<Edge> es;
    ns.reserve(nodes.size());
    es.reserve(edges.size());
    ns.push_back(nodes[root]);
    std::vector<int> old{root};
    for (size_t i = 0; i < old.size(); ++i) {
      auto n = nodes[old[i]];
      ns[i].begin = es.size();
      ns[i].pending = false;
      ns[i].pruned = false;
      for (int j = 0; j < n.count; ++j) {
        auto e = edges[n.begin + j];
        e.pruned = false;
        if (e.child >= 0) {
          old.push_back(e.child);
          ns.push_back(nodes[e.child]);
          e.child = int(ns.size() - 1);
        }
        es.push_back(e);
      }
    }
    nodes.swap(ns);
    edges.swap(es);
  }
};
struct Job {
  Tree *tree;
  int leaf;
  std::vector<int> path;
};
struct Engine {
  Config c;
  float time_fraction=-1; // Negative preserves fixed-simulation pruning.
  bool pending_loss_guard=false;
  Stats stats;
  std::vector<std::unique_ptr<Tree>> trees;
  std::vector<Job> jobs;
  std::string text;
  size_t cursor = 0;
  explicit Engine(Config cfg) : c(cfg) {
    if (c.simulations < 1 || c.batch < 1 || c.width < 1 || c.keep < 1 ||
        c.depth < 1 || c.depth > 512 || c.cpuct <= 0 || c.fpu < 0 ||
        c.widening <= 0 || c.prune <= 0 || c.prune > 1 ||
        !std::isfinite(c.cpuct) || !std::isfinite(c.fpu) ||
        !std::isfinite(c.widening) || !std::isfinite(c.prune))
      throw std::runtime_error("Invalid native MCTS configuration");
    jobs.reserve(c.batch);
  }
  void proof(Tree &t, int id) {
    auto &n = t.nodes[id];
    bool all = n.count > 0;
    int best = -1;
    for (int j = 0; j < n.count; ++j) {
      auto &e = t.edges[n.begin + j];
      int v = e.child < 0 ? Unknown : t.nodes[e.child].solved;
      if (v == -1) {
        n.solved = 1;
        return;
      }
      if (v == Unknown)
        all = false;
      else
        best = std::max(best, -v);
    }
    if (all)
      n.solved = best;
  }
  void backup(Job &job, float value) {
    auto &t = *job.tree;
    ++t.nodes[job.leaf].visits;
    // Recover parent IDs once, then reverse values and exact proofs together.
    std::array<int, 512> parents;
    int parent = 0, depth = 0;
    for (int ei : job.path) {
      parents[depth++] = parent;
      parent = t.edges[ei].child;
    }
    for (int k = depth - 1; k >= 0; --k) {
      value = -value;
      auto &e = t.edges[job.path[k]];
      ++e.visits;
      e.total += value;
      auto &n = t.nodes[parents[k]];
      ++n.visits;
      proof(t, parents[k]);
      if (n.solved != Unknown)
        value = float(n.solved);
    }
    ++t.done;
    ++stats.sims;
  }
  void prune(Tree &t) {
    auto &n = t.nodes[0];
    if (n.pruned || (time_fraction>=0 ? time_fraction<c.prune : t.done < int(std::ceil(c.simulations * c.prune))))
      return;
    n.pruned = true;
    std::vector<int> rank(n.count);
    std::iota(rank.begin(), rank.end(), 0);
    std::stable_sort(rank.begin(), rank.end(), [&](int a, int b) {
      auto &x = t.edges[n.begin + a];
      auto &y = t.edges[n.begin + b];
      if (x.visits != y.visits)
        return x.visits > y.visits;
      if (x.q() != y.q())
        return x.q() > y.q();
      return x.prior > y.prior;
    });
    std::vector<bool> keep(n.count, false);
    for (int j = 0; j < std::min(c.keep, n.count); ++j)
      keep[rank[j]] = true;
    for (int j = 0; j < n.count; ++j) {
      auto &e = t.edges[n.begin + j];
      if (j >= 4 && !keep[j] && !e.forcing &&
          !(e.child >= 0 && t.nodes[e.child].solved == -1)) {
        e.pruned = true;
        ++stats.pruned;
      }
    }
  }
  int select(Tree &t, int id) {
    auto &n = t.nodes[id];
    int width = c.width + int(c.widening * std::sqrt(n.visits)), seen = 0;
    bool safe = false;
    auto frontier = [&](const Edge &e, int rank) {
      return !e.pruned &&
             (rank < width || e.forcing ||
              (e.child >= 0 && t.nodes[e.child].solved == -1));
    };
    auto eligible = [&](const Edge &e,int rank) {
      return frontier(e,rank) && !(e.child>=0 && t.nodes[e.child].pending);
    };
    for (int j = 0; j < n.count; ++j) {
      auto &e = t.edges[n.begin + j];
      if (e.pruned)
        continue;
      int r = seen++;
      // Pending viable children still refute the claim that only known losses
      // remain. Flush the partial inference batch instead of burning the
      // simulation budget backing up a known loss while those children wait.
      if ((pending_loss_guard?frontier(e,r):eligible(e,r)) && (e.child < 0 || t.nodes[e.child].solved != 1))
        safe = true;
    }
    float best = -1e30f;
    int chosen = -1;
    seen = 0;
    float scale = c.cpuct * std::sqrt(std::max(1, n.visits));
    for (int j = 0; j < n.count; ++j) {
      int ei = n.begin + j;
      auto &e = t.edges[ei];
      if (e.pruned)
        continue;
      int r = seen++;
      if (!eligible(e, r))
        continue;
      if (safe && e.child >= 0 && t.nodes[e.child].solved == 1) {
        ++stats.skips;
        continue;
      }
      // Two virtual observations from the candidate Q/WDL estimate, decaying
      // as actual backed-up search evidence accumulates. Unsupervised actions
      // retain ordinary WDL first-play urgency.
      float q = std::isfinite(e.neural_q)
                    ? (e.total + 2.f * e.neural_q) / (e.visits + 2.f)
                    : (e.visits ? e.q() : std::max(-1.f, n.value - c.fpu));
      float score = q + scale * e.prior / (1 + e.visits);
      if (score > best) {
        best = score;
        chosen = ei;
      }
    }
    return chosen;
  }
  // Generate once, reuse the same edge block for all subsequent visits.
  void prepare(Tree &t, int id, Board &b, bool repeated) {
    Movelist ms;
    movegen::legalmoves(ms, b);
    int v = terminal(b, ms, repeated);
    if (v != Unknown) {
      t.nodes[id].solved = v;
      t.nodes[id].value = v;
      t.nodes[id].expanded = true;
      return;
    }
    if (ms.size() > Capacity)
      throw std::runtime_error("Legal move capacity exceeded");
    t.nodes[id].begin = t.edges.size();
    t.nodes[id].count = ms.size();
    for (auto m : ms) {
      Edge e;
      e.move = m;
      bool check = b.givesCheck(m) != CheckType::NO_CHECK;
      e.forcing = check || b.isCapture(m) || m.typeOf() == Move::PROMOTION;
      if (check) {
        b.makeMove(m);
        bool mate = b.inCheck() && !movegen::anylegalmoves(b);
        b.unmakeMove(m);
        if (mate) {
          e.child = t.node();
          t.nodes[e.child].solved = -1;
          t.nodes[e.child].expanded = true;
          t.nodes[e.child].value = -1;
          t.nodes[id].solved = 1;
          ++stats.mates;
        }
      }
      t.edges.push_back(e);
    }
  }
  int next(int capacity, uint8_t *tokens, float *meta, int16_t *ids,
           int *counts) {
    if (!jobs.empty())
      throw std::runtime_error("Submit pending evaluations first");
    int produced = 0, stalled = 0;
    while (produced < capacity && stalled < int(trees.size())) {
      Tree &t = *trees[cursor++ % trees.size()];
      if (t.done + t.inflight >= c.simulations ||
          t.nodes[0].solved != Unknown) {
        ++stalled;
        continue;
      }
      Board &b = t.work;
      struct Rewind {
        Board &b;
        std::array<Move, 512> moves;
        int count = 0;
        ~Rewind() {
          while (count)
            b.unmakeMove(moves[--count]);
        }
      } rewind{b, {}, 0};
      int ep = t.ep;
      bool repeated = t.repeated;
      int id = 0;
      Job job{&t, 0, {}};
      job.path.reserve(c.depth);
      prune(t);
      while (t.nodes[id].expanded && t.nodes[id].solved == Unknown &&
             int(job.path.size()) < c.depth) {
        int ei = select(t, id);
        if (ei < 0)
          break;
        auto m = t.edges[ei].move;
        ep = nextEp(b, m);
        b.makeMove(m);
        rewind.moves[rewind.count++] = m;
        repeated = repeated || b.isRepetition(1);
        if (t.edges[ei].child < 0) {
          int child = t.node();
          t.edges[ei].child = child;
        }
        job.path.push_back(ei);
        id = t.edges[ei].child;
      }
      job.leaf = id;
      stats.depth = std::max(stats.depth, uint64_t(job.path.size()));
      if (t.nodes[id].pending) {
        ++stalled;
        continue;
      }
      if (!t.nodes[id].expanded && t.nodes[id].count == 0)
        prepare(t, id, b, repeated);
      if (t.nodes[id].solved != Unknown) {
        if (job.path.empty()) {
          ++stalled;
          continue;
        }
        ++stats.terminals;
        backup(job, t.nodes[id].solved);
        stalled = 0;
        continue;
      }
      if (t.nodes[id].expanded) {
        if (int(job.path.size()) >= c.depth) {
          ++stats.limits;
          backup(job, t.nodes[id].value);
          stalled = 0;
          continue;
        }
        ++stalled;
        continue;
      }
      encode(b, ep, tokens + produced * 64, meta + produced * 12);
      auto &n = t.nodes[id];
      counts[produced] = n.count;
      std::fill(ids + produced * Capacity, ids + (produced + 1) * Capacity,
                int16_t(-1));
      for (int j = 0; j < n.count; ++j)
        ids[produced * Capacity + j] =
            action(t.edges[n.begin + j].move, b.sideToMove());
      n.pending = true;
      // Virtual loss reserves independent leaves within a tree without Python
      // callbacks.
      int parent = 0;
      for (int ei : job.path) {
        ++t.nodes[parent].visits;
        auto &e = t.edges[ei];
        ++e.visits;
        e.total -= 1;
        parent = e.child;
      }
      ++t.inflight;
      jobs.push_back(std::move(job));
      ++produced;
      stalled = 0;
    }
    if (produced)
      ++stats.batches;
    return produced;
  }
  void submit(const float *logits, const float *values, int count,
              const float *neural_q = nullptr) {
    if (count != int(jobs.size()))
      throw std::runtime_error("Batch size mismatch");
    // Restore ALL reservations before sorting edge blocks (paths store edge
    // indices).
    for (auto &job : jobs) {
      auto &t = *job.tree;
      int parent = 0;
      for (int ei : job.path) {
        --t.nodes[parent].visits;
        auto &e = t.edges[ei];
        --e.visits;
        e.total += 1;
        parent = e.child;
      }
    }
    for (int i = 0; i < count; ++i) {
      auto &job = jobs[i];
      auto &t = *job.tree;
      auto &n = t.nodes[job.leaf];
      float maximum = -INFINITY;
      for (int j = 0; j < n.count; ++j) {
        float v = logits[i * Capacity + j];
        if (!std::isfinite(v))
          throw std::runtime_error("Nonfinite policy");
        maximum = std::max(maximum, v);
      }
      if (!std::isfinite(values[i]))
        throw std::runtime_error("Nonfinite value");
      float sum = 0;
      for (int j = 0; j < n.count; ++j) {
        auto &e = t.edges[n.begin + j];
        if (neural_q) {
          float q = neural_q[i * Capacity + j];
          if (std::isinf(q) || (std::isfinite(q) && std::abs(q) > 1.f))
            throw std::runtime_error("Invalid candidate neural Q");
          e.neural_q = q;
        }
        e.prior = std::exp(logits[i * Capacity + j] - maximum);
        sum += e.prior;
      }
      float total = 0;
      for (int j = 0; j < n.count; ++j) {
        auto &e = t.edges[n.begin + j];
        e.prior = std::max(1e-8f, e.prior / sum);
        total += e.prior;
      }
      for (int j = 0; j < n.count; ++j)
        t.edges[n.begin + j].prior /= total;
      std::stable_sort(t.edges.begin() + n.begin,
                       t.edges.begin() + n.begin + n.count,
                       [](auto &a, auto &b) { return a.prior > b.prior; });
      n.value = std::clamp(values[i], -1.f, 1.f);
      n.expanded = true;
      n.pending = false;
      --t.inflight;
      ++stats.evals;
      if (!job.path.empty())
        backup(job, n.value);
    }
    jobs.clear();
  }
  int append(int cap,uint8_t *tokens,float *meta,int16_t *ids,int *counts,int *jobids) {
    std::vector<Job> existing;existing.swap(jobs);
    int n;
    try { n=next(cap,tokens,meta,ids,counts); }
    catch(...) {
      existing.insert(existing.end(),std::make_move_iterator(jobs.begin()),std::make_move_iterator(jobs.end()));
      jobs=std::move(existing);throw;
    }
    int offset=existing.size();
    for(int i=0;i<n;++i)jobids[i]=offset+i;
    existing.insert(existing.end(),std::make_move_iterator(jobs.begin()),std::make_move_iterator(jobs.end()));
    jobs=std::move(existing);return n;
  }
  void submit_some(const int *indices,const float *policy,const float *q,const float *values,int count) {
    if(count<1)throw std::runtime_error("Empty partial submission");
    std::vector<bool> selected(jobs.size(),false);
    for(int j=0;j<count;++j) {
      int i=indices[j];
      if(i<0 || i>=int(jobs.size()) || selected[i] || jobs[i].leaf<0)throw std::runtime_error("Invalid partial job ID");
      selected[i]=true;
      // Validate before moving outstanding jobs or restoring reservations.
      // A rejected network result must still be cancellable without losing
      // unrelated pending paths.
      const auto &job=jobs[i];const auto &node=job.tree->nodes[job.leaf];
      if(!std::isfinite(values[j]))throw std::runtime_error("Nonfinite value");
      for(int k=0;k<node.count;++k) {
        if(!std::isfinite(policy[j*Capacity+k]))throw std::runtime_error("Nonfinite policy");
        if(q && (std::isinf(q[j*Capacity+k]) || (std::isfinite(q[j*Capacity+k]) && std::abs(q[j*Capacity+k])>1.f)))
          throw std::runtime_error("Invalid candidate neural Q");
      }
    }
    std::vector<Job> waiting;waiting.swap(jobs);
    for(int j=0;j<count;++j)jobs.push_back(waiting[indices[j]]);
    submit(policy,values,count,q);
    for(int j=0;j<count;++j){waiting[indices[j]].leaf=-1;waiting[indices[j]].path.clear();}
    bool active=false;for(const auto &job:waiting)active=active || job.leaf>=0;
    if(active)jobs=std::move(waiting);
  }
  void cancel_pending() {
    for(auto &job:jobs) {
      if(job.leaf<0)continue;
      auto &t=*job.tree;int parent=0;
      for(int ei:job.path) {
        --t.nodes[parent].visits;auto &edge=t.edges[ei];--edge.visits;edge.total+=1;parent=edge.child;
      }
      t.nodes[job.leaf].pending=false;--t.inflight;
    }
    jobs.clear();
  }
};
thread_local std::string error;
template <class F> int protect(F f) {
  try {
    f();
    error.clear();
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
} // namespace
extern "C" {
const char *nc_error() { return error.c_str(); }
void *nc_create(Config c) {
  try {
    return new Engine(c);
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
void nc_destroy(void *p) { delete static_cast<Engine *>(p); }
int nc_time_progress(void *p,float fraction) {
  return protect([&] {
    if(!std::isfinite(fraction) || fraction < -1 || fraction>1)
      throw std::runtime_error("Invalid time progress");
    static_cast<Engine *>(p)->time_fraction=fraction;
  });
}
int nc_batch_safety(void *p,int enabled) {
  return protect([&] {
    if(enabled<0 || enabled>1)throw std::runtime_error("Invalid batch safety option");
    static_cast<Engine *>(p)->pending_loss_guard=enabled;
  });
}
int nc_forget(void *p) {
  return protect([&] {
    auto &e=*static_cast<Engine *>(p);
    if(!e.jobs.empty())throw std::runtime_error("Pending leaves at forget");
    e.trees.clear();e.cursor=0;
  });
}
int nc_start(void *p, int slot, const char *fen, const char *moves) {
  return protect([&] {
    auto &e = *static_cast<Engine *>(p);
    if (!e.jobs.empty())
      throw std::runtime_error("Pending leaves at start");
    std::string f(fen);
    std::istringstream in(moves);
    std::vector<std::string> hist;
    std::string s;
    while (in >> s)
      hist.push_back(s);
    while (int(e.trees.size()) <= slot)
      e.trees.push_back(nullptr);
    auto &ptr = e.trees[slot];
    bool reuse =
        ptr && ptr->initial == f && hist.size() >= ptr->history.size() &&
        std::equal(ptr->history.begin(), ptr->history.end(), hist.begin());
    int id = 0;
    if (reuse)
      for (size_t k = ptr->history.size(); k < hist.size(); ++k) {
        auto &n = ptr->nodes[id];
        int child = -1;
        for (int j = 0; j < n.count; ++j) {
          auto &edge = ptr->edges[n.begin + j];
          if (uci::moveToUci(edge.move) == hist[k]) {
            child = edge.child;
            break;
          }
        }
        if (child < 0) {
          reuse = false;
          break;
        }
        id = child;
      }
    if (reuse) {
      ptr->adopt(id);
      ++e.stats.reuse;
      e.stats.retained += ptr->nodes.size();
    } else
      ptr = std::make_unique<Tree>(f);
    auto &t = *ptr;
    // Advance only the newly played suffix when reusing; no replay of the
    // entire game and no duplicate legal generation for historical moves.
    size_t replay_from = reuse ? t.history.size() : 0;
    if (!reuse) {
      t.board = Board(f);
      t.ep = rawEp(f);
      t.repeated = false;
    }
    for (size_t k = replay_from; k < hist.size(); ++k) {
      auto &m = hist[k];
      auto move = uci::uciToMove(t.board, m);
      Movelist legal;
      movegen::legalmoves(legal, t.board);
      if (std::find(legal.begin(), legal.end(), move) == legal.end())
        throw std::runtime_error("Illegal root history");
      t.ep = nextEp(t.board, move);
      t.board.makeMove(move);
      t.repeated = t.repeated || t.board.isRepetition(1);
    }
    t.work = t.board;
    t.history = hist;
    t.done = 0;
    t.inflight = 0;
    t.base = t.nodes[0].visits;
    t.nodes[0].pruned = false;
    for (int j = 0; j < t.nodes[0].count; ++j)
      t.edges[t.nodes[0].begin + j].pruned = false;
    Movelist legal;
    movegen::legalmoves(legal, t.board);
    if (terminal(t.board, legal, t.repeated) != Unknown)
      throw std::runtime_error("Terminal root");
    ++e.stats.choices;
  });
}
int nc_resize(void *p, int n) {
  return protect([&] {
    auto &e = *static_cast<Engine *>(p);
    if (!e.jobs.empty() || n < 1)
      throw std::runtime_error("Invalid resize");
    e.trees.resize(n);
    e.cursor = 0;
  });
}
int nc_next(void *p, int cap, uint8_t *t, float *m, int16_t *a, int *c) {
  int n = 0;
  int rc =
      protect([&] { n = static_cast<Engine *>(p)->next(cap, t, m, a, c); });
  return rc < 0 ? -1 : n;
}
int nc_submit(void *p, const float *l, const float *v, int n) {
  return protect([&] { static_cast<Engine *>(p)->submit(l, v, n); });
}
int nc_submit_heads(void *p, const float *l, const float *q, const float *v, int n) {
  return protect([&] { static_cast<Engine *>(p)->submit(l, v, n, q); });
}
int nc_append(void *p,int cap,uint8_t *t,float *m,int16_t *a,int *c,int *jobids) {
  int n=0;int rc=protect([&]{n=static_cast<Engine *>(p)->append(cap,t,m,a,c,jobids);});return rc<0?-1:n;
}
int nc_submit_some(void *p,const int *jobs,const float *policy,const float *q,const float *values,int count) {
  return protect([&]{static_cast<Engine *>(p)->submit_some(jobs,policy,q,values,count);});
}
int nc_cancel_pending(void *p) {return protect([&]{static_cast<Engine *>(p)->cancel_pending();});}
int nc_root_ready(void *p) {
  auto &e=*static_cast<Engine *>(p);if(e.trees.empty() || !e.trees[0])return 0;
  const auto &n=e.trees[0]->nodes[0];return n.expanded || n.solved!=Unknown;
}
const char *nc_result(void *p) {
  auto &e = *static_cast<Engine *>(p);
  std::ostringstream out;
  out << "{\"stats\":{";
  auto &s = e.stats;
  out << "\"choices\":" << s.choices << ",\"leaf_evaluations\":" << s.evals
      << ",\"completed_simulations\":" << s.sims
      << ",\"terminal_leaves\":" << s.terminals
      << ",\"pruned_root_edges\":" << s.pruned
      << ",\"proven_loss_skips\":" << s.skips << ",\"mate_proofs\":" << s.mates
      << ",\"max_depth\":" << s.depth << ",\"depth_limit_hits\":" << s.limits
      << ",\"tree_reuses\":" << s.reuse << ",\"retained_nodes\":" << s.retained
      << ",\"inference_batches\":" << s.batches << "},\"roots\":[";
  bool comma = false;
  for (auto &ptr : e.trees) {
    auto &t = *ptr;
    auto &n = t.nodes[0];
    if (comma)
      out << ",";
    comma = true;
    int best = -1;
    bool safe = false;
    for (int j = 0; j < n.count; ++j) {
      auto &x = t.edges[n.begin + j];
      if (!x.pruned && (x.child < 0 || t.nodes[x.child].solved != 1))
        safe = true;
    }
    for (int j = 0; j < n.count; ++j) {
      int ei = n.begin + j;
      auto &x = t.edges[ei];
      int solved = x.child < 0 ? Unknown : t.nodes[x.child].solved;
      if (n.solved != Unknown ? solved != -n.solved
                              : (x.pruned || (safe && solved == 1)))
        continue;
      if (best < 0 || x.visits > t.edges[best].visits ||
          (x.visits == t.edges[best].visits &&
           (x.q() > t.edges[best].q() ||
            (x.q() == t.edges[best].q() && x.prior > t.edges[best].prior))))
        best = ei;
    }
    out << "{\"move\":\""
        << (best < 0 ? "0000" : uci::moveToUci(t.edges[best].move))
        << "\",\"simulations\":" << n.visits
        << ",\"new_simulations\":" << t.done
        << ",\"retained_visits\":" << t.base << ",\"nodes\":" << t.nodes.size()
        << ",\"proven_value\":"
        << (n.solved == Unknown ? "null" : std::to_string(n.solved))
        << ",\"leaders\":[";
    std::vector<int> ranked(n.count);
    std::iota(ranked.begin(), ranked.end(), n.begin);
    std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) {
      return t.edges[a].visits > t.edges[b].visits;
    });
    for (int k = 0; k < std::min(5, n.count); ++k) {
      auto &x = t.edges[ranked[k]];
      if (k)
        out << ",";
      out << "{\"move\":\"" << uci::moveToUci(x.move)
          << "\",\"visits\":" << x.visits << ",\"q\":" << x.q()
          << ",\"prior\":" << x.prior
          << ",\"pruned\":" << (x.pruned ? "true" : "false") << "}";
    }
    out << "]}";
  }
  out << "]}";
  e.text = out.str();
  return e.text.c_str();
}
// Differential-test interface: exercise the exact same native rules/encoding as
// search.
int nc_probe(const char *fen, const char *moves, uint8_t *tokens, float *meta,
             int16_t *ids, int *count, int *value) {
  return protect([&] {
    Board b(fen);
    int ep = rawEp(fen);
    bool repeated = false;
    std::istringstream in(moves);
    std::string s;
    while (in >> s) {
      auto m = uci::uciToMove(b, s);
      ep = nextEp(b, m);
      b.makeMove(m);
      repeated = repeated || b.isRepetition(1);
    }
    Movelist ms;
    movegen::legalmoves(ms, b);
    encode(b, ep, tokens, meta);
    *count = ms.size();
    for (int i = 0; i < *count; ++i)
      ids[i] = action(ms[i], b.sideToMove());
    *value = terminal(b, ms, repeated);
  });
}
}
