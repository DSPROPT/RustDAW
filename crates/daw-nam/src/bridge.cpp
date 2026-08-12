#include <cmath>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/container.h"
#include "NAM/convnet.h"
#include "NAM/linear.h"
#include "NAM/lstm.h"
#include "NAM/wavenet/model.h"
#include "NAM/wavenet/slimmable.h"

struct RustDawNam {
  std::unique_ptr<nam::DSP> dsp;
  std::vector<NAM_SAMPLE> samples;
};

static thread_local std::string last_error;

// Referencing each translation unit prevents a static-library linker from
// discarding the files whose initializers register NAM architecture parsers.
static void retain_architecture_registrars() {
  volatile auto container = &nam::container::create_config;
  volatile auto convnet = &nam::convnet::create_config;
  volatile auto linear = &nam::linear::create_config;
  volatile auto lstm = &nam::lstm::create_config;
  volatile auto wavenet = &nam::wavenet::create_config;
  volatile auto slimmable = &nam::slimmable_wavenet::create_config;
  (void)container; (void)convnet; (void)linear; (void)lstm; (void)wavenet; (void)slimmable;
}

extern "C" {
RustDawNam* rustdaw_nam_load(const char* path, double sample_rate, int max_block) {
  try {
    retain_architecture_registrars();
    auto model = nam::get_dsp(std::filesystem::path(path));
    const double expected = model->GetExpectedSampleRate();
    if (expected > 0.0 && std::abs(expected - sample_rate) > 0.5) {
      last_error = "NAM model expects " + std::to_string(expected) +
                   " Hz, but the session uses " + std::to_string(sample_rate) + " Hz";
      return nullptr;
    }
    model->Reset(sample_rate, max_block);
    return new RustDawNam{std::move(model), std::vector<NAM_SAMPLE>(max_block)};
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "unknown error while loading NAM model";
  }
  return nullptr;
}

void rustdaw_nam_free(RustDawNam* model) { delete model; }

// A model trained with a known loudness carries it, so captures can be levelled
// against each other. Not every model has one; the caller has to cope.
bool rustdaw_nam_loudness(RustDawNam* model, double* out) {
  if (model == nullptr || out == nullptr || !model->dsp->HasLoudness()) return false;
  try {
    *out = model->dsp->GetLoudness();
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "unknown error while reading NAM loudness";
  }
  return false;
}

bool rustdaw_nam_process(RustDawNam* model, float* samples, int frames) {
  if (model == nullptr || samples == nullptr || frames <= 0) return false;
  if (static_cast<size_t>(frames) > model->samples.size()) {
    last_error = "audio block exceeds the prepared NAM block size";
    return false;
  }
  try {
    for (int i = 0; i < frames; ++i) model->samples[i] = samples[i];
    NAM_SAMPLE* channels[] = {model->samples.data()};
    model->dsp->process(channels, channels, frames);
    for (int i = 0; i < frames; ++i) samples[i] = static_cast<float>(model->samples[i]);
    return true;
  } catch (const std::exception& error) {
    last_error = error.what();
  } catch (...) {
    last_error = "unknown error while processing NAM model";
  }
  return false;
}

const char* rustdaw_nam_last_error() { return last_error.c_str(); }
}
