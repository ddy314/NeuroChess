#include "../neurochess/native/mcts_pipeline.cpp"
#include <cassert>
#include <iostream>

int main() {
  Config c{512, 8, 6, 12, 128, 1.5f, .2f, 1.5f, .75f};
  Engine e(c);
  e.pending_loss_guard = true;
  assert(nc_start(&e, 0, constants::STARTPOS, "") == 0);
  uint8_t tokens[8 * 64];
  float meta[8 * 12], policy[8 * 256]{}, values[8]{};
  int16_t ids[8 * 256];
  int counts[8], jobs[8];
  int n = e.append(8, tokens, meta, ids, counts, jobs);
  assert(n == 1);
  e.submit_some(jobs, policy, nullptr, values, n);
  auto &t = *e.trees[0];
  n = e.append(8, tokens, meta, ids, counts, jobs);
  int before = t.inflight;
  values[0] = NAN;
  bool rejected = false;
  try { e.submit_some(jobs, policy, nullptr, values, 1); }
  catch (const std::runtime_error &) { rejected = true; }
  assert(rejected && t.inflight == before && int(e.jobs.size()) == n);
  values[0] = 0;
  e.cancel_pending();
  assert(t.inflight == 0 && t.nodes[0].visits == t.done);
  for (int round = 0; round < 30; ++round) {
    n = e.append(8, tokens, meta, ids, counts, jobs);
    assert(n > 0);
    // Complete a non-prefix subset, then append without reassigning the
    // outstanding job IDs. This exercises delayed cache misses and OOO hits.
    int last = jobs[n - 1];
    e.submit_some(&last, policy, nullptr, values, 1);
    e.append(3, tokens, meta, ids, counts, jobs);
    e.cancel_pending();
    assert(e.jobs.empty() && t.inflight == 0);
    assert(t.nodes[0].visits == t.done);
    for (const auto &node : t.nodes) {
      assert(!node.pending && node.visits >= 0);
      int edge_visits = 0;
      for (int j = 0; j < node.count; ++j) {
        const auto &edge = t.edges[node.begin + j];
        assert(edge.visits >= 0 && std::abs(edge.total) <= edge.visits + 1e-5);
        edge_visits += edge.visits;
      }
      assert(edge_visits <= node.visits);
    }
  }
  // After cancellation, the exact remaining simulation budget is still usable.
  while ((n = e.append(8, tokens, meta, ids, counts, jobs)))
    e.submit_some(jobs, policy, nullptr, values, n);
  assert(t.done == c.simulations && t.inflight == 0);
  assert(t.nodes[0].visits == c.simulations);
  // Faster inference makes searches larger than the original vector reserves.
  // Exercise both reallocations under UBSan, not just the small batch path.
  c.simulations = 8192;
  Engine large(c);
  large.pending_loss_guard = true;
  assert(nc_start(&large, 0, constants::STARTPOS, "") == 0);
  while ((n = large.append(8, tokens, meta, ids, counts, jobs)))
    large.submit_some(jobs, policy, nullptr, values, n);
  auto &lt = *large.trees[0];
  assert(lt.nodes.size() > 4096 && lt.edges.size() > 131072);
  assert(lt.done == 8192 && lt.inflight == 0 && lt.nodes[0].visits == 8192);
  std::cout << "partial backup, cancellation, virtual loss and budget invariants passed\n";
}
