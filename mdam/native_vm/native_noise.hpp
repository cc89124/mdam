// native_noise.hpp — native port of ttn_backend.frame_layer.ClifftNoiseSampler + _apply_noise_site
// (list-channel sites) + _apply_mask_words.  Reproduces clifft's gap-based noise firing schedule and
// its RNG consumption EXACTLY (init draw + per-fire channel draw + advance gap draw).
//
// Static data (hazards = cumsum(-log1p(-prob)); per-site channel masks) is compiled ONCE in Python
// and passed in; runtime does only RNG draws + searchsorted + frame mask application.
#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include "native_rng.hpp"
#include "native_frame.hpp"
#include "native_instr.hpp"

namespace mdam {

// one channel of a list-type noise site: probability + little-endian Pauli mask words
struct NoiseChannel { double prob; std::vector<uint64_t> x_words, z_words; };
struct NoiseSite { std::vector<NoiseChannel> channels; };   // list-type site

struct NativeNoiseSampler {
    std::vector<double> hazards;        // cumsum(-log1p(-prob)), shot-static
    int next_idx = 0;
    NativeRng* rng = nullptr;
    uint64_t draws = 0;                 // RNG draw counter (for verification)
    bool log_on = false;                // §0 internal-trace: record fired (site, x_word, z_word)
    std::vector<std::array<uint64_t,3>> fire_log;
    bool noapply = false;               // Gate J 2B: skip the frame mask application (compiled away);
                                        // RNG draws + fire_log are UNCHANGED.  default false = unchanged.

    void init(const std::vector<double>& hz, NativeRng* r) {
        hazards = hz; rng = r; next_idx = 0; draws = 0; draw_next();
    }
    inline double udraw() { draws++; return rng->next_double(); }
    void draw_next() {
        if (hazards.empty() || next_idx >= (int)hazards.size()) { next_idx = -1; return; }
        double cur = (next_idx == 0) ? 0.0 : hazards[next_idx - 1];
        double gap = -std::log1p(-udraw());
        double key = cur + gap;
        // numpy searchsorted(hazards, key, side='right') == upper_bound
        next_idx = (int)(std::upper_bound(hazards.begin(), hazards.end(), key) - hazards.begin());
    }
    bool should_fire(int site) const { return site == next_idx; }
    void advance() { next_idx++; draw_next(); }

    // apply a list-type noise site if it fires (== _apply_noise_site list branch)
    void apply_site(int site_idx, const NoiseSite& site, NativeFrame& frame) {
        if (!should_fire(site_idx)) return;
        double prob_sum = 0.0; for (auto& c : site.channels) prob_sum += c.prob;
        if (prob_sum <= 0.0) { advance(); return; }
        double u = udraw() * prob_sum;
        double cum = 0.0; size_t ci = 0;
        for (auto& c : site.channels) {
            cum += c.prob;
            if (u < cum) {
                if (!noapply) ISKIP(ISK_NOISE_APPLY, {
                for (size_t wi = 0; wi < c.x_words.size(); wi++) { uint64_t w = c.x_words[wi];
                    while (w) { int b = __builtin_ctzll(w); w &= w-1; frame.apply_x((uint32_t)(wi*64+b)); } }
                for (size_t wi = 0; wi < c.z_words.size(); wi++) { uint64_t w = c.z_words[wi];
                    while (w) { int b = __builtin_ctzll(w); w &= w-1; frame.apply_z((uint32_t)(wi*64+b)); } } });
                // Record the CHANNEL INDEX (not word[0] of the Pauli): word[0] cannot uniquely identify a
                // multi-word channel (qubits > 64 live in higher words) -> the compiled/shadow paths matched
                // the wrong channel -> wrong dynbit.  noise_base[site]+ci is exactly compile_jprogram's mapping.
                if (log_on) fire_log.push_back({(uint64_t)site_idx, (uint64_t)ci, 0});
                break;
            }
            ci++;
        }
        advance();
    }
};

} // namespace mdam
