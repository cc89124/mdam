// Stage B numerical kernel: the in-place dense Pauli linear combination
//   phi <- alpha*phi + bph*(P phi),   P = X^mx Z^mz on the magic register, bph = beta * i^pp.
// BIT-IDENTICAL to nearclifford_backend.clifft_axis.engine._pauli_lincomb_inplace's full-formula
// paths (the SCALAR / vectorized diagonal and off-diagonal branches) -- it does NOT replace the
// Step-1 "diaghalf" global-phase fast path (that stays in Python).  phi is interleaved complex128
// (phi[2j]=re, phi[2j+1]=im).  mx,mz are r-bit masks over the magic register (r small -> uint64).
// Convention (matches _apply_magic_pauli): (P phi)[j] = (-1)^par((j^? )&mz) phi[j^mx]; the exact
// per-branch signs below mirror the Python kernel line-for-line.  No RNG, no allocation.
#include <cstdint>

extern "C" {

// off-diagonal (mx != 0) butterfly:  pair (j, k=j^mx), j over the half with pivot bit clear.
//   phi[j] = alpha*a + bph*(sk*b);  phi[k] = alpha*b + bph*(sj*a)
//   sj = 1-2*par(j&mz);  sk = 1-2*par(k&mz)
void lincomb_offdiag(double* phi, int64_t N, uint64_t mx, uint64_t mz,
                     double ar, double ai, double br, double bi) {
    const uint64_t pivot = mx & (~mx + 1ULL);          // lowest set bit of mx
    for (int64_t j = 0; j < N; ++j) {
        if (j & pivot) continue;
        int64_t k = j ^ (int64_t)mx;
        double a_re = phi[2*j],   a_im = phi[2*j+1];
        double b_re = phi[2*k],   b_im = phi[2*k+1];
        int sj = 1 - 2 * __builtin_parityll((uint64_t)j & mz);
        int sk = 1 - 2 * __builtin_parityll((uint64_t)k & mz);
        double skb_re = sk * b_re, skb_im = sk * b_im;
        double sja_re = sj * a_re, sja_im = sj * a_im;
        // phi[j] = alpha*a + bph*(sk*b)
        phi[2*j]   = (ar*a_re - ai*a_im) + (br*skb_re - bi*skb_im);
        phi[2*j+1] = (ar*a_im + ai*a_re) + (br*skb_im + bi*skb_re);
        // phi[k] = alpha*b + bph*(sj*a)
        phi[2*k]   = (ar*b_re - ai*b_im) + (br*sja_re - bi*sja_im);
        phi[2*k+1] = (ar*b_im + ai*b_re) + (br*sja_im + bi*sja_re);
    }
}

// diagonal (mx == 0, mz != 0) full formula:  phi[s] *= (alpha + bph) if par(s&mz)==0 else (alpha - bph)
void lincomb_diag(double* phi, int64_t N, uint64_t mz,
                  double ar, double ai, double br, double bi) {
    const double er = ar + br, ei = ai + bi;           // m_even = alpha + bph
    const double oR = ar - br, oI = ai - bi;            // m_odd  = alpha - bph
    for (int64_t s = 0; s < N; ++s) {
        int par = __builtin_parityll((uint64_t)s & mz);
        double mr = par ? oR : er, mi = par ? oI : ei;
        double re = phi[2*s], im = phi[2*s+1];
        phi[2*s]   = re*mr - im*mi;
        phi[2*s+1] = re*mi + im*mr;
    }
}

}  // extern "C"
