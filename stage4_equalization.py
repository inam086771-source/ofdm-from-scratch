# -*- coding: utf-8 -*-
"""
OFDM From Scratch — Stage 4

Equalization: undo the per-subcarrier channel distortion from Stage 3.

Key OFDM insight: thanks to the cyclic prefix, linear convolution with
the channel becomes CIRCULAR convolution over one OFDM symbol. Circular
convolution in the time domain = simple multiplication in the frequency
domain. So instead of needing to "deconvolve" a difficult time-domain
mess, each subcarrier k just got multiplied by a single complex number
H[k] (the channel's frequency response at that subcarrier). To undo it,
we simply DIVIDE each received subcarrier by H[k]. This is the entire
reason OFDM is so much easier to equalize than single-carrier systems.

This stage uses PERFECT channel knowledge (H[k] computed directly from
the true channel taps) to verify the equalization math itself is
correct, before Stage 5 replaces this with a realistic *estimated* H[k]
from pilot symbols.

Run this file directly: depends on stage1_2_ofdm_modulation.py and
stage3_channel.py being in the same folder.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage1_2_ofdm_modulation import (
    bits_to_qpsk, qpsk_to_bits, ofdm_modulate, ofdm_demodulate,
)
from stage3_channel import multipath_channel, add_awgn


def compute_channel_frequency_response(taps, n_subcarriers):
    """Convert time-domain channel taps -> per-subcarrier frequency
    response H[k], by zero-padding the taps to n_subcarriers length and
    taking the FFT (same transform direction as the receiver's FFT, so
    the multiplication lines up correctly per subcarrier)."""
    taps_padded = np.zeros(n_subcarriers, dtype=complex)
    taps_padded[:len(taps)] = taps
    H = np.fft.fft(taps_padded)
    return H


def equalize(rx_freq_symbols, H, n_subcarriers):
    """Divide each received subcarrier by the channel's frequency
    response at that subcarrier -- this is zero-forcing equalization,
    the simplest form (ignores noise amplification at deep fades, which
    is a known limitation revisited later if needed)."""
    n_ofdm_symbols = len(rx_freq_symbols) // n_subcarriers
    rx_blocks = rx_freq_symbols.reshape(n_ofdm_symbols, n_subcarriers)
    eq_blocks = rx_blocks / H[np.newaxis, :]
    return eq_blocks.flatten()


if __name__ == "__main__":
    np.random.seed(42)

    N_SUBCARRIERS = 64
    CP_LEN = 16
    N_OFDM_SYMBOLS = 20
    SNR_DB = 15

    n_bits = N_SUBCARRIERS * N_OFDM_SYMBOLS * 2
    tx_bits = np.random.randint(0, 2, n_bits)
    tx_symbols = bits_to_qpsk(tx_bits)
    tx_signal = ofdm_modulate(tx_symbols, N_SUBCARRIERS, CP_LEN)

    rx_signal_multipath, taps = multipath_channel(tx_signal, n_taps=5, seed=1)
    rx_signal = add_awgn(rx_signal_multipath, SNR_DB)

    rx_symbols_noeq = ofdm_demodulate(rx_signal, N_SUBCARRIERS, CP_LEN)
    evm_noeq = np.sqrt(np.mean(np.abs(rx_symbols_noeq - tx_symbols) ** 2))

    # --- Equalize using PERFECT channel knowledge ---
    H = compute_channel_frequency_response(taps, N_SUBCARRIERS)
    rx_symbols_eq = equalize(rx_symbols_noeq, H, N_SUBCARRIERS)
    evm_eq = np.sqrt(np.mean(np.abs(rx_symbols_eq - tx_symbols) ** 2))

    print(f"EVM WITHOUT equalization: {evm_noeq:.3f}")
    print(f"EVM WITH equalization (perfect channel knowledge): {evm_eq:.3f}")
    print(f"Improvement: {evm_noeq / evm_eq:.1f}x lower EVM")

    # --- Decode bits and check BER ---
    rx_bits_noeq = qpsk_to_bits(rx_symbols_noeq)
    rx_bits_eq = qpsk_to_bits(rx_symbols_eq)

    ber_noeq = np.mean(rx_bits_noeq != tx_bits)
    ber_eq = np.mean(rx_bits_eq != tx_bits)

    print(f"\nBit Error Rate WITHOUT equalization: {ber_noeq:.4f} "
          f"({int(ber_noeq * n_bits)}/{n_bits} bits wrong)")
    print(f"Bit Error Rate WITH equalization:    {ber_eq:.4f} "
          f"({int(ber_eq * n_bits)}/{n_bits} bits wrong)")

    if ber_eq < ber_noeq:
        print("\n✅ PASS: equalization reduced the bit error rate, as expected.")
    else:
        print("\n❌ Unexpected: equalization did not improve BER -- check the math.")

    # --- Plot: 3-way constellation comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].scatter(tx_symbols.real, tx_symbols.imag, alpha=0.6)
    axes[0].set_title('Transmitted')

    axes[1].scatter(rx_symbols_noeq.real, rx_symbols_noeq.imag,
                      alpha=0.4, color='red', s=15)
    axes[1].set_title(f'Received, NO equalization\nEVM={evm_noeq:.2f}, '
                        f'BER={ber_noeq:.3f}')

    axes[2].scatter(rx_symbols_eq.real, rx_symbols_eq.imag,
                      alpha=0.4, color='green', s=15)
    axes[2].set_title(f'Received, WITH equalization\n'
                        f'(perfect H), EVM={evm_eq:.2f}, BER={ber_eq:.3f}')

    for ax in axes:
        ax.set_xlabel('In-phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('stage4_equalization.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot: stage4_equalization.png")
    print("\nNext: Stage 5 will replace perfect channel knowledge with a "
          "REALISTIC estimate of H[k], using pilot symbols -- the way "
          "real receivers actually do it (they don't know H in advance).")
