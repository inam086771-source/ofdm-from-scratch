# -*- coding: utf-8 -*-
"""
OFDM From Scratch — Stage 5

Realistic channel estimation using PILOT symbols, replacing Stage 4's
"cheat" (perfect knowledge of the true channel taps).

How real receivers actually do this: the transmitter and receiver agree
in advance on a set of KNOWN symbols (pilots), sent on specific
subcarriers. Since the receiver knows what was sent on those
subcarriers, it can directly measure H[k] = received / transmitted at
exactly those pilot positions. For the DATA subcarriers (where the
transmitted value is unknown -- that's the whole point, we're trying to
recover it), the receiver INTERPOLATES the channel estimate from
nearby pilots.

This introduces a real, unavoidable estimation error (the receiver
never sees the *exact* H[k] at data subcarriers, only an interpolated
guess) -- which is why Stage 5's BER will be slightly worse than
Stage 4's perfect-knowledge BER. That gap IS the point: it's the
realistic cost of not knowing the channel in advance.

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
from stage4_equalization import compute_channel_frequency_response, equalize


def insert_pilots(data_symbols, n_subcarriers, pilot_spacing, pilot_value=1 + 0j):
    """Build one OFDM symbol's worth of subcarriers, inserting a known
    pilot symbol every `pilot_spacing` subcarriers, with data symbols
    filling the rest.

    Returns:
        full_block: n_subcarriers complex values (pilots + data)
        pilot_indices: which subcarrier indices are pilots
        data_indices: which subcarrier indices carry actual data
    """
    pilot_indices = np.arange(0, n_subcarriers, pilot_spacing)
    data_indices = np.array([i for i in range(n_subcarriers) if i not in pilot_indices])
    assert len(data_indices) == len(data_symbols), \
        f"Expected {len(data_indices)} data symbols, got {len(data_symbols)}"

    full_block = np.zeros(n_subcarriers, dtype=complex)
    full_block[pilot_indices] = pilot_value
    full_block[data_indices] = data_symbols
    return full_block, pilot_indices, data_indices


def estimate_channel_from_pilots(rx_block, pilot_indices, pilot_value,
                                   n_subcarriers):
    """Least-squares channel estimate at pilot positions, then linear
    interpolation across all subcarriers (including data positions)."""
    H_at_pilots = rx_block[pilot_indices] / pilot_value

    # Interpolate real and imaginary parts separately across ALL
    # subcarrier indices, using the known pilot positions as anchors.
    all_indices = np.arange(n_subcarriers)
    H_real = np.interp(all_indices, pilot_indices, H_at_pilots.real)
    H_imag = np.interp(all_indices, pilot_indices, H_at_pilots.imag)
    H_estimated = H_real + 1j * H_imag
    return H_estimated


if __name__ == "__main__":
    np.random.seed(42)

    N_SUBCARRIERS = 64
    CP_LEN = 16
    PILOT_SPACING = 8   # every 8th subcarrier is a pilot -> 8 pilots, 56 data
    N_OFDM_SYMBOLS = 20
    SNR_DB = 15

    pilot_indices = np.arange(0, N_SUBCARRIERS, PILOT_SPACING)
    n_data_per_symbol = N_SUBCARRIERS - len(pilot_indices)
    print(f"{len(pilot_indices)} pilot subcarriers, "
          f"{n_data_per_symbol} data subcarriers per OFDM symbol")

    # --- Build TX: interleave pilots + data across N_OFDM_SYMBOLS blocks ---
    n_data_bits = n_data_per_symbol * N_OFDM_SYMBOLS * 2
    tx_bits = np.random.randint(0, 2, n_data_bits)
    tx_data_symbols = bits_to_qpsk(tx_bits)  # length = n_data_per_symbol * N_OFDM_SYMBOLS

    tx_blocks = []
    for i in range(N_OFDM_SYMBOLS):
        chunk = tx_data_symbols[i * n_data_per_symbol:(i + 1) * n_data_per_symbol]
        full_block, p_idx, d_idx = insert_pilots(chunk, N_SUBCARRIERS, PILOT_SPACING)
        tx_blocks.append(full_block)
    tx_symbols_full = np.concatenate(tx_blocks)  # includes pilots now

    tx_signal = ofdm_modulate(tx_symbols_full, N_SUBCARRIERS, CP_LEN)

    # --- Channel + noise (same as Stage 3/4) ---
    rx_signal_multipath, taps = multipath_channel(tx_signal, n_taps=5, seed=1)
    rx_signal = add_awgn(rx_signal_multipath, SNR_DB)
    rx_symbols_full = ofdm_demodulate(rx_signal, N_SUBCARRIERS, CP_LEN)

    # --- True H (for comparison only -- not used in the estimation itself) ---
    H_true = compute_channel_frequency_response(taps, N_SUBCARRIERS)

    # --- Per-OFDM-symbol: estimate H from pilots, equalize, extract data ---
    rx_blocks = rx_symbols_full.reshape(N_OFDM_SYMBOLS, N_SUBCARRIERS)
    rx_data_estimated = []
    H_estimates = []

    for i in range(N_OFDM_SYMBOLS):
        block = rx_blocks[i]
        H_est = estimate_channel_from_pilots(block, pilot_indices, 1 + 0j, N_SUBCARRIERS)
        H_estimates.append(H_est)
        eq_block = block / H_est
        rx_data_estimated.append(eq_block[d_idx])  # d_idx same every block

    rx_data_estimated = np.concatenate(rx_data_estimated)

    # --- Compare against Stage 4's perfect-knowledge equalization ---
    rx_data_perfect = []
    for i in range(N_OFDM_SYMBOLS):
        block = rx_blocks[i]
        eq_block = block / H_true
        rx_data_perfect.append(eq_block[d_idx])
    rx_data_perfect = np.concatenate(rx_data_perfect)

    evm_estimated = np.sqrt(np.mean(np.abs(rx_data_estimated - tx_data_symbols) ** 2))
    evm_perfect = np.sqrt(np.mean(np.abs(rx_data_perfect - tx_data_symbols) ** 2))

    rx_bits_estimated = qpsk_to_bits(rx_data_estimated)
    rx_bits_perfect = qpsk_to_bits(rx_data_perfect)
    ber_estimated = np.mean(rx_bits_estimated != tx_bits)
    ber_perfect = np.mean(rx_bits_perfect != tx_bits)

    print(f"\nEVM with PERFECT channel knowledge:        {evm_perfect:.3f}")
    print(f"EVM with PILOT-based channel ESTIMATION:    {evm_estimated:.3f}")
    print(f"\nBER with PERFECT channel knowledge:         {ber_perfect:.4f}")
    print(f"BER with PILOT-based channel ESTIMATION:    {ber_estimated:.4f}")

    if evm_estimated >= evm_perfect:
        print("\n✅ PASS: as expected, pilot-based estimation is close to but "
              "slightly worse than perfect knowledge (realistic estimation "
              "error, not a bug).")
    else:
        print("\n⚠️  Estimated EVM beat perfect knowledge -- likely a fluke "
              "of this particular noise realization, worth re-checking "
              "with a different seed.")

    # --- Plot: H_true vs H_estimated (magnitude across subcarriers) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(np.abs(H_true), 'g-', label='True |H[k]|', linewidth=2)
    axes[0].plot(np.abs(H_estimates[0]), 'r--', label='Estimated |H[k]| (1 symbol)')
    axes[0].scatter(pilot_indices, np.abs(H_true[pilot_indices]),
                      color='black', zorder=5, label='Pilot positions')
    axes[0].set_title('Channel frequency response: true vs. pilot-estimated')
    axes[0].set_xlabel('Subcarrier index')
    axes[0].set_ylabel('|H[k]|')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(rx_data_estimated.real, rx_data_estimated.imag,
                      alpha=0.4, color='orange', s=15)
    axes[1].set_title(f'Data constellation, pilot-based equalization\n'
                        f'EVM={evm_estimated:.2f}, BER={ber_estimated:.3f}')
    axes[1].set_xlabel('In-phase (I)')
    axes[1].set_ylabel('Quadrature (Q)')
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].axvline(0, color='gray', linewidth=0.5)
    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].grid(alpha=0.3)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('stage5_pilot_estimation.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot: stage5_pilot_estimation.png")
    print("\nNext: Stage 6 will sweep SNR and plot BER vs SNR, comparing "
          "against the theoretical QPSK BER curve -- the standard way "
          "wireless simulation results are reported.")
