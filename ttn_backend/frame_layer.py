"""
frame_layer.py -- Clifft frame layer + constant-pool helpers.

This module previously contained a dense reference simulator (DenseState class)
that has been deprecated in favor of clifft.sample() as the authoritative
ground truth. Only the frame layer and constant-pool helpers remain, which are
shared between Clifft validation and the TTN backend's frame layer.

Public exports used by ttn_backend.py:
    PauliFrame          -- frame layer (X/Z parity bits)
    ClifftNoiseSampler  -- per-shot OP_NOISE hazard-gap firing schedule
    _apply_cp_mask      -- OP_APPLY_PAULI cp_mask application
    _apply_noise_site   -- OP_NOISE / OP_NOISE_BLOCK site application
    _bits               -- iterate set bits in integer masks
    _d                  -- inst.as_dict() with fallback
    IGNORE_OPS          -- ops to skip in dispatch
    FLAG_SIGN           -- measurement flag bit
    T_PHASE, T_PHASE_DAG -- T-gate phases
"""
from __future__ import annotations
import math
import numpy as np
import clifft

INV_SQRT2 = 1.0 / math.sqrt(2.0)
T_PHASE = complex(math.cos(math.pi/4),  math.sin(math.pi/4))
T_PHASE_DAG = complex(math.cos(math.pi/4), -math.sin(math.pi/4))

FLAG_SIGN = int(getattr(clifft, "FLAG_SIGN", 1))

IGNORE_OPS = {"OP_DETECTOR", "OP_POSTSELECT", "OP_OBSERVABLE", "OP_EXP_VAL"}


def _d(inst):
    """Get inst.as_dict() safely."""
    try: return inst.as_dict()
    except Exception: return {}


def _bits(mask):
    """Yield set bit indices of an integer mask, least-significant first."""
    mask = int(mask)
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


# ---------- constant pool access ----------

# Attribute name fallback lists. After Codex's bindings.cc patch:
#   prog.pauli_masks  -- exposed
#   prog.noise_sites  -- exposed
# Earlier names kept as fallback in case of revert / different builds.
CP_MASK_ATTRS = ('pauli_masks', 'cp_masks', 'cp_pauli_masks',
                 'apply_pauli_masks', 'masks',
                 'const_pool_masks', 'constant_pool_masks')

CP_NOISE_ATTRS = ('noise_sites', 'cp_noise_sites', 'noises',
                  'cp_noises', 'noise_pool')


def _cp_get(prog, attr_candidates, idx):
    """Try multiple attribute names on prog; return prog.<attr>[idx] or None."""
    for a in attr_candidates:
        v = getattr(prog, a, None)
        if v is None: continue
        try:
            return v[idx]
        except Exception:
            continue
    return None


def _apply_mask_words(frame, x_words, z_words):
    """Apply little-endian uint64 Pauli mask words to the Python frame."""
    for wi, word in enumerate(x_words):
        word = int(word)
        base = wi * 64
        while word:
            low = word & -word
            frame.apply_x(base + low.bit_length() - 1)
            word ^= low
    for wi, word in enumerate(z_words):
        word = int(word)
        base = wi * 64
        while word:
            low = word & -word
            frame.apply_z(base + low.bit_length() - 1)
            word ^= low


# ---------- Pauli frame ----------

class PauliFrame:
    """X/Z parity bit array, with Clifford-style frame updates."""
    def __init__(self, n=256):
        self.x = np.zeros(n, dtype=np.uint8)
        self.z = np.zeros(n, dtype=np.uint8)
        self.n = n
    def _g(self, s):
        if s >= self.n:
            new_n = max(self.n*2, s+1)
            nx = np.zeros(new_n, dtype=np.uint8); nx[:self.n] = self.x
            nz = np.zeros(new_n, dtype=np.uint8); nz[:self.n] = self.z
            self.x, self.z, self.n = nx, nz, new_n
    def h(self, s):     self._g(s); self.x[s], self.z[s] = self.z[s], self.x[s]
    def s_gate(self, s):self._g(s); self.z[s] ^= self.x[s]
    def cnot(self,c,t): self._g(max(c,t)); self.x[t]^=self.x[c]; self.z[c]^=self.z[t]
    def cz(self,a,b):   self._g(max(a,b)); self.z[a]^=self.x[b]; self.z[b]^=self.x[a]
    def swap(self,a,b):
        self._g(max(a,b))
        self.x[a],self.x[b]=self.x[b],self.x[a]
        self.z[a],self.z[b]=self.z[b],self.z[a]
    def apply_x(self,s): self._g(s); self.x[s]^=1
    def apply_z(self,s): self._g(s); self.z[s]^=1
    def apply_y(self,s): self._g(s); self.x[s]^=1; self.z[s]^=1
    def set_xz(self, s, x, z=0):
        self._g(s)
        self.x[s] = int(x) & 1
        self.z[s] = int(z) & 1
    def xb(self,s): return int(self.x[s]) if s<self.n else 0
    def zb(self,s): return int(self.z[s]) if s<self.n else 0


# ---------- Constant-pool aware helpers ----------

def _apply_cp_mask(prog, mask_idx, frame, rng):
    """Apply Pauli mask from constant pool to frame."""
    m = _cp_get(prog, CP_MASK_ATTRS, mask_idx)
    if m is None:
        return
    if isinstance(m, dict) and 'x_words' in m and 'z_words' in m:
        _apply_mask_words(frame, m['x_words'], m['z_words'])
        return
    if isinstance(m, dict):
        slots = m.get('slots', m.get('targets', []))
        paulis = m.get('paulis', m.get('pauli', []))
        for s, p in zip(slots, paulis):
            if int(p) & 1: frame.apply_x(int(s))
            if int(p) & 2: frame.apply_z(int(s))
        return
    try:
        for item in m:
            try:
                s, p = item
                s = int(s); p = int(p)
                if p & 1: frame.apply_x(s)
                if p & 2: frame.apply_z(s)
            except Exception:
                pass
        return
    except Exception:
        pass


def _noise_probabilities(prog):
    sites = getattr(prog, 'noise_sites', None)
    probs = getattr(prog, 'noise_site_probabilities', None)
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        if sites is not None:
            probs = probs[:len(sites)]
        return probs
    if sites is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray([sum(float(ch.get('prob', 0.0)) for ch in site) for site in sites],
                      dtype=np.float64)


class ClifftNoiseSampler:
    """Mirror clifft.sample's gap-based OP_NOISE firing schedule for one shot."""
    def __init__(self, prog, rng):
        probs = _noise_probabilities(prog)
        probs = probs[np.isfinite(probs)]
        probs = np.clip(probs, 0.0, 1.0 - 2.0**-53)
        self.hazards = np.cumsum(-np.log1p(-probs))
        self.next_noise_idx = 0
        self.rng = rng
        self.draw_next_noise()

    def draw_next_noise(self):
        if self.hazards.size == 0 or self.next_noise_idx >= self.hazards.size:
            self.next_noise_idx = -1
            return
        current = 0.0 if self.next_noise_idx == 0 else float(self.hazards[self.next_noise_idx - 1])
        gap = -math.log1p(-float(self.rng.random()))
        self.next_noise_idx = int(np.searchsorted(self.hazards, current + gap, side='right'))

    def should_fire(self, site_idx):
        return int(site_idx) == self.next_noise_idx

    def advance_after_fire(self):
        self.next_noise_idx += 1
        self.draw_next_noise()


def _apply_noise_site(prog, site_idx, frame, rng, noise_sampler=None):
    """Apply a Clifft noise site if the shot-level scheduler says it fires."""
    if noise_sampler is not None and not noise_sampler.should_fire(site_idx):
        return
    site = _cp_get(prog, CP_NOISE_ATTRS, site_idx)
    if site is None:
        return
    if isinstance(site, list):
        prob_sum = sum(float(ch.get('prob', 0.0)) for ch in site)
        if prob_sum <= 0.0:
            if noise_sampler is not None:
                noise_sampler.advance_after_fire()
            return
        u = float(rng.random()) * prob_sum
        cumulative = 0.0
        for ch in site:
            cumulative += float(ch.get('prob', 0.0))
            if u < cumulative:
                _apply_mask_words(frame, ch.get('x_words', ()), ch.get('z_words', ()))
                break
        if noise_sampler is not None:
            noise_sampler.advance_after_fire()
        return
    if isinstance(site, dict):
        slots = site.get('sites', site.get('slots', site.get('targets', [])))
        pX = float(site.get('pX', site.get('p_x', 0.0)))
        pY = float(site.get('pY', site.get('p_y', 0.0)))
        pZ = float(site.get('pZ', site.get('p_z', 0.0)))
        p_dep = float(site.get('p', site.get('probability', 0.0)))
        if p_dep > 0 and pX + pY + pZ == 0:
            pX = pY = pZ = p_dep / 3
        for s in slots:
            u = rng.random()
            if u < pX:        frame.apply_x(int(s))
            elif u < pX+pY:   frame.apply_y(int(s))
            elif u < pX+pY+pZ: frame.apply_z(int(s))
        return
