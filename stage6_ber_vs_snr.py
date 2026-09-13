# -*- coding: utf-8 -*-
"""
OFDM From Scratch — Stage 6 (final stage)

Sweep SNR and plot BER vs SNR -- the standard way wireless simulation
results are reported and validated.

Two curves, for two different reasons:

1. AWGN-only (no multipath), our OFDM chain vs. the THEORETICAL QPSK
   BER formula. This is a correctness check: an AWGN-only channel has a
   known closed-form answer, so if our simulated curve matches theory,
   that's strong evidence the whole OFDM chain (modulation, IFFT/FFT,
   CP, demod) is implemented correctly -- the same role the "reach the
   target BLER" validation played in the URLLC project.

2. Full pipeline (multipath + AWGN + REALISTIC pilot-based channel
   estimation), which will sit noticeably above the theoretical/AWGN
   curve -- showing the real-world cost of a fading channel plus
   imperfect channel knowledge, not a bug.

Run this file directly: depends on all previous stage files being in
the same folder. This sweep trains nothing (no ML here, unlike the
URLLC project) but runs many trials per SNR point, so it can take
a minute or two.
"""

import numpy as np
from scipy.special import erfc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage1_2_ofdm_modulation import bits_to_qpsk, qpsk_to_bits, ofdm_modulate, ofdm_demodulate
from stage3_channel import multipath_channel, add_awgn
from stage4_equalization import compute_channel_frequency_response
from stage5_pilot_estimation import insert_pilots, estimate_channel_from_pilots


def theoretical_qpsk_ber(snr_db):
    """Closed-form BER for QPSK over an AWGN channel (Gray-coded, so
    each bit error is independent): BER = Q(sqrt(2*Eb/N0)).

    Our simulation's SNR is defined as Es/N0 (signal power = symbol
    energy, since bits_to_qpsk gives unit-energy symbols). For QPSK,
    Es = 2*Eb (2 bits carried per symbol), so 2*Eb/N0 = Es/N0 = SNR,
    giving BER = Q(sqrt(SNR)) = 0.5*erfc(sqrt(SNR)/sqrt(2))
                              = 0.5*erfc(sqrt(SNR/2)).
    (An earlier version of this function used 0.5*erfc(sqrt(SNR)),
    missing the /2 -- that made the theoretical curve too optimistic
    and was caught by comparing against the simulated AWGN-only curve,
    which sat clearly above it at every SNR point.)
    """
    snr_linear = 10 ** (snr_db / 10)
    return 0.5 * erfc(np.sqrt(snr_linear / 2))


def run_awgn_only_trial(n_bits, snr_db, n_subcarriers, cp_len, seed):
    """AWGN-only OFDM trial (no multipath) -- used only for the
    correctness check against theory."""
    rng = np.random.default_rng(seed)
    tx_bits = rng.integers(0, 2, n_bits)
    tx_symbols = bits_to_qpsk(tx_bits)
    tx_signal = ofdm_modulate(tx_symbols, n_subcarriers, cp_len)

    signal_power = np.mean(np.abs(tx_signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(len(tx_signal)) + 1j * rng.standard_normal(len(tx_signal))
    )
    rx_signal = tx_signal + noise

    rx_symbols = ofdm_demodulate(rx_signal, n_subcarriers, cp_len)
    rx_bits = qpsk_to_bits(rx_symbols)
    return np.mean(rx_bits != tx_bits)


def run_full_pipeline_trial(n_ofdm_symbols, snr_db, n_subcarriers, cp_len,
                              pilot_spacing, channel_seed, data_seed):
    """Full pipeline: multipath + AWGN + pilot-based channel estimation."""
    pilot_indices = np.arange(0, n_subcarriers, pilot_spacing)
    d_idx = np.array([i for i in range(n_subcarriers) if i not in pilot_indices])
    n_data_per_symbol = len(d_idx)

    rng = np.random.default_rng(data_seed)
    n_data_bits = n_data_per_symbol * n_ofdm_symbols * 2
    tx_bits = rng.integers(0, 2, n_data_bits)
    tx_data_symbols = bits_to_qpsk(tx_bits)

    tx_blocks = []
    for i in range(n_ofdm_symbols):
        chunk = tx_data_symbols[i * n_data_per_symbol:(i + 1) * n_data_per_symbol]
        full_block, _, _ = insert_pilots(chunk, n_subcarriers, pilot_spacing)
        tx_blocks.append(full_block)
    tx_symbols_full = np.concatenate(tx_blocks)
    tx_signal = ofdm_modulate(tx_symbols_full, n_subcarriers, cp_len)

    rx_signal_multipath, taps = multipath_channel(tx_signal, n_taps=5, seed=channel_seed)
    rx_signal = add_awgn(rx_signal_multipath, snr_db)
    rx_symbols_full = ofdm_demodulate(rx_signal, n_subcarriers, cp_len)
    rx_blocks = rx_symbols_full.reshape(n_ofdm_symbols, n_subcarriers)

    rx_data = []
    for i in range(n_ofdm_symbols):
        block = rx_blocks[i]
        H_est = estimate_channel_from_pilots(block, pilot_indices, 1 + 0j, n_subcarriers)
        eq_block = block / H_est
        rx_data.append(eq_block[d_idx])
    rx_data = np.concatenate(rx_data)

    rx_bits = qpsk_to_bits(rx_data)
    return np.mean(rx_bits != tx_bits)


if __name__ == "__main__":
    N_SUBCARRIERS = 64
    CP_LEN = 16
    PILOT_SPACING = 8
    SNR_RANGE_DB = np.arange(0, 16, 2)
    N_OFDM_SYMBOLS_PER_TRIAL = 20

    # More trials at higher SNR: BER gets much lower there, so more bits
    # are needed to reliably observe enough errors to estimate it (the
    # same logic as the URLLC project's Monte Carlo averaging -- low
    # trial counts under-sample rare-error regimes and bias the
    # measured BER upward).
    def trials_for_snr(snr_db):
        if snr_db <= 6:
            return 20
        elif snr_db <= 10:
            return 100
        else:
            return 500

    ber_awgn_sim = []
    ber_full_pipeline = []

    n_bits_awgn = N_SUBCARRIERS * N_OFDM_SYMBOLS_PER_TRIAL * 2

    for snr_db in SNR_RANGE_DB:
        n_trials = trials_for_snr(snr_db)
        awgn_trials = [run_awgn_only_trial(n_bits_awgn, snr_db, N_SUBCARRIERS, CP_LEN,
                                             seed=1000 + snr_db * 1000 + t)
                       for t in range(n_trials)]
        full_trials = [run_full_pipeline_trial(N_OFDM_SYMBOLS_PER_TRIAL, snr_db,
                                                 N_SUBCARRIERS, CP_LEN, PILOT_SPACING,
                                                 channel_seed=t, data_seed=2000 + snr_db * 1000 + t)
                       for t in range(n_trials)]

        ber_awgn_sim.append(np.mean(awgn_trials))
        ber_full_pipeline.append(np.mean(full_trials))

        print(f"SNR={snr_db:>3d}dB ({n_trials:>3d} trials)  |  "
              f"AWGN-only (sim): {ber_awgn_sim[-1]:.2e}  |  "
              f"Full pipeline: {ber_full_pipeline[-1]:.2e}")

    ber_theory = theoretical_qpsk_ber(SNR_RANGE_DB)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(SNR_RANGE_DB, ber_theory, 'k-', linewidth=2,
                 label='Theoretical QPSK BER (AWGN)')
    ax.semilogy(SNR_RANGE_DB, ber_awgn_sim, 'go', markersize=8,
                 label='Simulated OFDM, AWGN-only (correctness check)')
    ax.semilogy(SNR_RANGE_DB, ber_full_pipeline, 'r^-', markersize=8,
                 label='Full pipeline: multipath + pilot-based estimation')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bit Error Rate (BER)')
    ax.set_title('OFDM BER vs SNR: correctness check + realistic performance')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('stage6_ber_vs_snr.png', dpi=150, bbox_inches='tight')

    print("\nSaved plot: stage6_ber_vs_snr.png")
    print("\nIf the green dots (AWGN-only simulation) sit closely on the "
          "black theoretical line, that validates the whole OFDM "
          "implementation (Stages 1-2) is mathematically correct. The red "
          "curve sitting above both shows the realistic performance cost "
          "of a fading channel + imperfect (pilot-based) channel "
          "knowledge, which is the expected, correct outcome -- not an "
          "error.")
