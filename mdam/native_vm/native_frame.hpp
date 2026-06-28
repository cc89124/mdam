// native_frame.hpp — native port of ttn_backend.frame_layer.PauliFrame (dormant Clifford layer).
// X/Z parity bit arrays + Clifford frame updates, bit-identical to the Python implementation.
#pragma once
#include <cstdint>
#include <vector>

namespace mdam {

struct NativeFrame {
    std::vector<uint8_t> x, z;   // parity bits (0/1), index = slot

    explicit NativeFrame(size_t n = 256) : x(n, 0), z(n, 0) {}

    inline void grow(size_t s) {
        if (s >= x.size()) {
            size_t nn = x.size() * 2; if (nn < s + 1) nn = s + 1;
            x.resize(nn, 0); z.resize(nn, 0);
        }
    }
    inline void h(uint32_t s)        { grow(s); uint8_t t = x[s]; x[s] = z[s]; z[s] = t; }
    inline void s_gate(uint32_t s)   { grow(s); z[s] ^= x[s]; }
    inline void cnot(uint32_t c, uint32_t t) { grow(c > t ? c : t); x[t] ^= x[c]; z[c] ^= z[t]; }
    inline void cz(uint32_t a, uint32_t b)   { grow(a > b ? a : b); z[a] ^= x[b]; z[b] ^= x[a]; }
    inline void swap(uint32_t a, uint32_t b) { grow(a > b ? a : b);
        uint8_t t = x[a]; x[a] = x[b]; x[b] = t; t = z[a]; z[a] = z[b]; z[b] = t; }
    inline void apply_x(uint32_t s)  { grow(s); x[s] ^= 1; }
    inline void apply_z(uint32_t s)  { grow(s); z[s] ^= 1; }
    inline void apply_y(uint32_t s)  { grow(s); x[s] ^= 1; z[s] ^= 1; }
    inline void set_xz(uint32_t s, uint8_t xx, uint8_t zz) { grow(s); x[s] = xx & 1; z[s] = zz & 1; }
    inline int  xb(uint32_t s) const { return s < x.size() ? (int)x[s] : 0; }
    inline int  zb(uint32_t s) const { return s < z.size() ? (int)z[s] : 0; }
};

} // namespace mdam
