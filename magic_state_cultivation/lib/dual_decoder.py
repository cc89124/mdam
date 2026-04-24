"""Y_L decoder wrapper + Pauli frame tracking for analytical fidelity.

Wraps the paper's CompiledDesaturationSampler for Y_L decoding, and
extracts OBSERVABLE_INCLUDE tracking indices for frame correction.

The per-shot correction is:
    Y_corr = raw_Y * static_sign_Y * (-1)**(frame_Y ^ pred_Y)

where frame_Y is the absolute parity of Y_L tracking measurements,
and pred_Y is the decoder's Y_L flip prediction. Measurement errors
on tracking bits cancel algebraically.
"""

from __future__ import annotations

import numpy as np
import sinter
import stim

from cultiv._decoding._desaturation_sampler import CompiledDesaturationSampler


def extract_frame_tracking_indices(circuit: stim.Circuit) -> list[int]:
    """Extract Y_L frame tracking measurement indices.

    Parses the OBSERVABLE_INCLUDE chain and returns the union of all
    referenced measurement record absolute indices, excluding the
    final OBSERVABLE_INCLUDE (the destructive Y_L MPP measurement).

    Uses the stim API for robust measurement counting.
    """
    abs_meas_idx = 0
    obs_groups: list[list[int]] = []

    for instr in circuit.flattened():
        name = instr.name

        # Count measurements using stim's own logic
        temp = stim.Circuit()
        temp.append(instr)
        num_m = temp.num_measurements

        if num_m == 0:
            if name == "OBSERVABLE_INCLUDE":
                group = []
                for t in instr.targets_copy():
                    if t.is_measurement_record_target:
                        group.append(abs_meas_idx + t.value)
                obs_groups.append(group)
            continue

        abs_meas_idx += num_m

    # Exclude the final OBSERVABLE_INCLUDE (destructive Y_L measurement)
    tracking_groups = obs_groups[:-1]
    return [idx for group in tracking_groups for idx in group]


class DecoderWithFrameTracking:
    """Wraps the paper's Y_L decoder with Pauli frame tracking.

    The paper's CompiledDesaturationSampler decodes Y_L observable
    flips. The frame tracking corrects the random Pauli frame from
    escape-stage lattice surgery measurements.
    """

    def __init__(
        self,
        s_proxy_circuit: stim.Circuit,
        noise_strength: float,
    ):
        """
        Args:
            s_proxy_circuit: The noisy Y-basis S-proxy stim circuit.
            noise_strength: Physical noise level (for metadata).
        """
        # Build the paper's decoder
        dem = s_proxy_circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
            ignore_decomposition_failures=True,
        )
        task = sinter.Task(
            circuit=s_proxy_circuit,
            decoder="desat",
            detector_error_model=dem,
            json_metadata={"p": noise_strength},
        )
        self.decoder = CompiledDesaturationSampler.from_task(task)

        # Extract frame tracking indices
        self.track_YL = extract_frame_tracking_indices(s_proxy_circuit)

        # Store postselected detector set and physical detector count
        self.postselected = self.decoder.postselected_detectors
        self.num_physical_dets = s_proxy_circuit.num_detectors
        self.num_gap_dets = self.decoder.num_dets

        # Precompute postselection discard mask (bit-packed)
        num_bytes = (self.num_physical_dets + 7) // 8
        self._discard_mask = np.zeros(num_bytes, dtype=np.uint8)
        for d in self.postselected:
            if d < self.num_physical_dets:
                self._discard_mask[d // 8] |= 1 << (d % 8)

    def postselect(self, detectors: np.ndarray) -> np.ndarray:
        """Return boolean mask of shots surviving postselection.

        Args:
            detectors: uint8 array (n_shots, num_physical_dets).

        Returns:
            Boolean array (n_shots,), True = keep.
        """
        dets_packed = np.packbits(
            detectors[:, : self.num_physical_dets].astype(np.uint8),
            axis=1,
            bitorder="little",
        )
        fired = np.any(dets_packed & self._discard_mask, axis=1)
        return ~fired

    def decode_and_correct(
        self,
        detectors: np.ndarray,
        measurements: np.ndarray,
        raw_Y: np.ndarray,
        static_sign_Y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode and correct Y_L expectation values.

        Args:
            detectors: uint8 (n_shots, num_physical_dets)
            measurements: uint8 (n_shots, num_measurements)
            raw_Y: float64 (n_shots,) — raw EXP_VAL Y_L values
            static_sign_Y: sign calibration from noiseless baseline

        Returns:
            (Y_corrected, gaps): both shape (n_shots,)
        """
        n_shots = detectors.shape[0]
        if n_shots == 0:
            return np.zeros(0), np.zeros(0)

        # Frame tracking: absolute parity of Y_L tracking measurements
        if self.track_YL:
            frame_Y = np.bitwise_xor.reduce(
                measurements[:, self.track_YL], axis=1
            ).astype(bool)
        else:
            frame_Y = np.zeros(n_shots, dtype=bool)

        # Pad detectors for virtual pair nodes + observable detector
        padded = np.zeros((n_shots, self.num_gap_dets), dtype=np.uint8)
        padded[:, : self.num_physical_dets] = detectors
        dets_packed = np.ascontiguousarray(
            np.packbits(padded, axis=1, bitorder="little"),
            dtype=np.uint8,
        )

        # Decode using paper's two-pass gap method
        predictions, gaps = self.decoder._decode_batch_overwrite_last_byte(
            dets_packed
        )
        pred_Y = predictions.reshape(-1).astype(bool)

        # Correct: measurement errors cancel in frame ^ pred
        Y_corrected = raw_Y * static_sign_Y * (-1.0) ** (frame_Y ^ pred_Y)

        return Y_corrected, gaps.reshape(-1)


def build_decoder(
    dcolor: int,
    noise_strength: float,
) -> DecoderWithFrameTracking:
    """Build a DecoderWithFrameTracking from circuit parameters."""
    from cultiv import make_end2end_cultivation_circuit
    from gen import NoiseModel

    r_growing = dcolor
    circuit = make_end2end_cultivation_circuit(
        dcolor=dcolor,
        dsurface=15,
        basis="Y",
        r_growing=r_growing,
        r_end=5,
        inject_style="unitary",
    )
    noisy = NoiseModel.uniform_depolarizing(
        noise_strength
    ).noisy_circuit_skipping_mpp_boundaries(circuit)

    return DecoderWithFrameTracking(noisy, noise_strength)


def calibrate_static_sign(
    circuit_text: str,
    track_YL: list[int],
) -> tuple[float, np.ndarray]:
    """Calibrate the static Y_L sign from a noiseless baseline shot.

    Returns:
        (static_sign_Y, ideal_measurements)
    """
    import clifft

    hir_nl = clifft.HirPassManager()
    hir_nl.add(clifft.RemoveNoisePass())
    prog_nl = clifft.compile(
        circuit_text,
        hir_passes=hir_nl,
        bytecode_passes=clifft.default_bytecode_pass_manager(),
    )
    ideal = clifft.sample(prog_nl, shots=1, seed=0)
    ideal_meas = ideal.measurements[0]

    ideal_frame_Y = sum(int(ideal_meas[i]) for i in track_YL) % 2
    raw_Y = ideal.exp_vals[0, 1]

    if abs(raw_Y) < 1e-4:
        static_sign_Y = 1.0
    else:
        static_sign_Y = float(np.sign(raw_Y * (-1) ** ideal_frame_Y))

    return static_sign_Y, ideal_meas
