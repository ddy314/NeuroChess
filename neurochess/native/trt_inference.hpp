#pragma once
#include <cstdint>
#include <memory>
#include <string>

namespace nc_native {
// Sequential API. Returned pinned host buffers remain valid until next call.
// Packed fused output: policy[256], prior-Q[256], value, support count, changed.
class TrtInference {
public:
  TrtInference(const std::string &plan, int maximum_batch = 8, bool graphs = true);
  ~TrtInference();
  TrtInference(const TrtInference &) = delete;
  TrtInference &operator=(const TrtInference &) = delete;
  const float *evaluate(const uint8_t *tokens, const float *meta,
                        const int16_t *moves, int batch);
  const float *raw_output(int batch); // after evaluate; policy,Q,WDL probabilities
  int maximum_batch() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace nc_native

// QA bridge only. The standalone executable never loads Python or this bridge.
extern "C" {
void *nci_create(const char *plan, int batch, int graphs);
void nci_destroy(void *handle);
int nci_evaluate(void *handle, const uint8_t *t, const float *m,
                 const int16_t *ids, int batch, float *fused, float *raw);
const char *nci_error();
int nci_fuse_device(const float *policy, const float *q, const float *wdl,
    const int32_t *ids, float *fused, float *raw, int batch, uintptr_t stream);
}
