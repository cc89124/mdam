// native_record.hpp — native port of the backend measurement record (backend.py self.record).
//
// The authoritative record is a dict {classical_idx -> bit} flattened by sample() into
// out[sh, cidx] for 0<=cidx<num_measurements.  The faithful native equivalent is a dense
// uint8 buffer of capacity = num_measurements, preallocated once and reset (not reallocated)
// per shot; reads default to 0 (== dict.get(cidx, 0)); out-of-range writes/reads are no-ops/0
// (matches sample()'s 0<=cidx<num_measurements filter and .get's absent-default).
//
// NOTE: this MDAM backend's sample() path returns ONLY the measurement record; it does NOT
// evaluate stim-style detectors/observables (no such evaluation exists in run_shot).  So the
// native record buffer ports exactly that model; detector/observable parity is not part of
// this backend's measurement path and is intentionally not invented here.
#pragma once
#include <cstdint>
#include <vector>

namespace mdam {

struct NativeRecordBuffer {
    std::vector<uint8_t> bits;     // dense, index = classical_idx
    uint32_t cap = 0;

    void init(uint32_t num_measurements) { cap = num_measurements; bits.assign(cap, 0); }
    void reset() { std::fill(bits.begin(), bits.end(), 0); }    // no realloc

    inline void set(uint32_t cidx, int bit) { if (cidx < cap) bits[cidx] = (uint8_t)(bit & 1); }
    inline int  get(uint32_t cidx) const { return cidx < cap ? (int)bits[cidx] : 0; }  // dict.get(.,0)
    inline void flip(uint32_t cidx) { if (cidx < cap) bits[cidx] ^= 1; }               // readout noise
};

} // namespace mdam
