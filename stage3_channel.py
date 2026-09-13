# -*- coding: utf-8 -*-
"""
OFDM From Scratch — Stage 3

Push the Stage 1+2 OFDM signal through a wireless channel: multipath
(Rician-flavored, consistent with the URLLC project) + AWGN noise.

At this stage we do NOT try to correct for the channel yet -- the point
is to see and measure the distortion it introduces, which is exactly
why equalization (Stage 4) will be necessary.

Run this file directly: it depends on stage1_2_ofdm_modulation.py being
in the same folder.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage1_2_ofdm_modulation import (
    bits_to_qpsk, qpsk_to_bits, ofdm_modulate, ofdm_demodulate,
)


# ==========================================
# CHANNEL MODEL
# ==========================================

def multipath_channel(tx_signal, n_taps=5, seed=None):
    """A simple static multipath channel: n_taps random complex tap
    coefficients (a basic tapped-delay-line model), applied via linear
    convolution.

    This is a simpler stand-in than the full Jakes/Rician Doppler model
    from the URLLC project (that one models fading *over time* for a
    mobile UE; here we just want a static multipath channel to
    demonstrate why OFDM needs a cyclic prefix + equalization).

    Rician-flavored: tap 0 (the direct/LOS path) is made much stronger
    than the other (NLOS/reflected) taps, controlled by the same 7dB
    K-factor used in the URLLC project, for consistency.
    """
    rng = np.random.default_rng(seed)
    K_linear = 10 ** (7 / 10)  # same 7 dB K-factor as the URLLC project

    taps = (rng.normal(size=n_taps) + 1j * rng.normal(size=n_taps)) / np.sqrt(2)
    taps[0] = np.sqrt(K_linear / (K_linear + 1)) * (1 + 0j)
    taps[1:] *= np.sqrt(1 / (K_linear + 1)) / np.sqrt(n_taps - 1)

    rx_signal = np.convolve(tx_signal, taps, mode='full')[:len(tx_signal)]
    return rx_signal, taps


def add_awgn(signal, snr_db):
    """Add complex additive white Gaussian noise at the given SNR (dB),
    relative to the average power of `signal`."""
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    return signal + noise


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)

    N_SUBCARRIERS = 64
    CP_LEN = 16
    N_OFDM_SYMBOLS = 20
    SNR_DB = 15  # deliberately not too high, so distortion is visible

    n_bits = N_SUBCARRIERS * N_OFDM_SYMBOLS * 2
    tx_bits = np.random.randint(0, 2, n_bits)
    tx_symbols = bits_to_qpsk(tx_bits)
    tx_signal = ofdm_modulate(tx_symbols, N_SUBCARRIERS, CP_LEN)

    print(f"TX signal ready: {len(tx_signal)} samples, "
          f"{N_OFDM_SYMBOLS} OFDM symbols, SNR={SNR_DB} dB")

    # --- Push through multipath channel ---
    rx_signal_multipath, taps = multipath_channel(tx_signal, n_taps=5, seed=1)
    print(f"Applied {len(taps)}-tap multipath channel "
          f"(tap magnitudes: {np.round(np.abs(taps), 3)})")

    # --- Add AWGN noise ---
    rx_signal = add_awgn(rx_signal_multipath, SNR_DB)

    # --- Demodulate WITHOUT any equalization (naive) ---
    rx_symbols_noeq = ofdm_demodulate(rx_signal, N_SUBCARRIERS, CP_LEN)

    # --- Quick sanity metric: how far off are we without equalization? ---
    evm = np.sqrt(np.mean(np.abs(rx_symbols_noeq - tx_symbols) ** 2))
    print(f"\nError Vector Magnitude (EVM) without equalization: {evm:.3f}")
    print("(Expect this to be large / constellation visibly rotated & "
          "scattered -- multipath rotates+scales each subcarrier "
          "differently, noise adds scatter. This is exactly the "
          "distortion Stage 4's equalizer needs to undo.)")

    # --- Plot: constellation comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].scatter(tx_symbols.real, tx_symbols.imag, alpha=0.6)
    axes[0].set_title('Transmitted QPSK constellation')
    axes[0].set_xlabel('In-phase (I)')
    axes[0].set_ylabel('Quadrature (Q)')
    axes[0].axhline(0, color='gray', linewidth=0.5)
    axes[0].axvline(0, color='gray', linewidth=0.5)
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[0].grid(alpha=0.3)
    axes[0].set_aspect('equal')

    axes[1].scatter(rx_symbols_noeq.real, rx_symbols_noeq.imag,
                      alpha=0.4, color='red', s=15)
    axes[1].set_title(f'Received (multipath + {SNR_DB}dB noise,\n'
                        f'NO equalization) -- EVM={evm:.2f}')
    axes[1].set_xlabel('In-phase (I)')
    axes[1].set_ylabel('Quadrature (Q)')
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].axvline(0, color='gray', linewidth=0.5)
    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].grid(alpha=0.3)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('stage3_channel_distortion.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot: stage3_channel_distortion.png")
    print("\nNext: Stage 4 will estimate/undo this per-subcarrier "
          "distortion (equalization) to recover a clean constellation.")
