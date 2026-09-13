# -*- coding: utf-8 -*-
"""
OFDM From Scratch — Stage 1 & 2

Stage 1: Bits -> QPSK symbols (digital modulation)
Stage 2: Symbols -> OFDM time-domain signal (IFFT + cyclic prefix)
         + perfect-reconstruction test (no channel yet -- should recover
         the exact original symbols, since there's no noise or multipath
         at this stage).

Run this file directly to see the constellation diagram and confirm
perfect reconstruction.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; remove this line if running with a display
import matplotlib.pyplot as plt


# ==========================================
# STAGE 1: Digital modulation (QPSK)
# ==========================================

# QPSK maps every 2 bits to one complex symbol. Using Gray coding so that
# adjacent constellation points differ by only 1 bit (minimizes bit errors
# when noise causes a symbol to be mistaken for its nearest neighbor).
QPSK_MAP = {
    (0, 0): (1 + 1j) / np.sqrt(2),
    (0, 1): (-1 + 1j) / np.sqrt(2),
    (1, 1): (-1 - 1j) / np.sqrt(2),
    (1, 0): (1 - 1j) / np.sqrt(2),
}
QPSK_DEMAP = {v: k for k, v in QPSK_MAP.items()}


def bits_to_qpsk(bits):
    """bits: 1D array of 0/1, length must be even.
    Returns: 1D complex array of QPSK symbols, length = len(bits) / 2."""
    assert len(bits) % 2 == 0, "QPSK needs an even number of bits"
    bits = np.array(bits).reshape(-1, 2)
    symbols = np.array([QPSK_MAP[tuple(pair)] for pair in bits])
    return symbols


def qpsk_to_bits(symbols):
    """Inverse of bits_to_qpsk -- nearest-neighbor detection.
    (Used later once we have real, noisy symbols; for the perfect-channel
    case, symbols already exactly match a constellation point.)"""
    constellation_points = np.array(list(QPSK_DEMAP.keys()))
    bits = []
    for s in symbols:
        nearest = constellation_points[np.argmin(np.abs(constellation_points - s))]
        bits.extend(QPSK_DEMAP[nearest])
    return np.array(bits)


# ==========================================
# STAGE 2: OFDM signal generation
# ==========================================

def ofdm_modulate(symbols, n_subcarriers, cp_len):
    """Arrange `symbols` onto `n_subcarriers`, take IFFT to go from
    frequency domain -> time domain, then prepend a cyclic prefix.

    symbols: 1D complex array, length must be a multiple of n_subcarriers
    n_subcarriers: number of OFDM subcarriers per symbol block
    cp_len: cyclic prefix length (samples copied from the end of each
            block to its front)

    Returns: 1D complex time-domain signal (all OFDM symbols concatenated).
    """
    assert len(symbols) % n_subcarriers == 0, \
        "symbol count must be a multiple of n_subcarriers"
    n_ofdm_symbols = len(symbols) // n_subcarriers
    freq_blocks = symbols.reshape(n_ofdm_symbols, n_subcarriers)

    # IFFT: frequency-domain subcarrier values -> time-domain waveform.
    # This is the mathematical heart of OFDM -- it's what lets many
    # narrowband subcarriers be transmitted simultaneously as one
    # wideband time-domain signal, all orthogonal to each other.
    time_blocks = np.fft.ifft(freq_blocks, axis=1)

    # Cyclic prefix: copy the LAST cp_len samples of each block to its
    # front. This does two things: (1) turns linear convolution with the
    # channel into effectively circular convolution, which is what makes
    # the simple per-subcarrier equalization in Stage 4 work, and
    # (2) absorbs multipath delay spread so one OFDM symbol doesn't smear
    # into the next (inter-symbol interference).
    cp = time_blocks[:, -cp_len:]
    with_cp = np.hstack([cp, time_blocks])

    return with_cp.flatten()


def ofdm_demodulate(rx_signal, n_subcarriers, cp_len):
    """Inverse of ofdm_modulate, assuming perfect timing (no channel yet).
    Strip the cyclic prefix from each block, then FFT back to frequency
    domain to recover the original per-subcarrier symbols."""
    symbol_len = n_subcarriers + cp_len
    n_ofdm_symbols = len(rx_signal) // symbol_len
    blocks = rx_signal.reshape(n_ofdm_symbols, symbol_len)

    # Strip cyclic prefix
    time_blocks = blocks[:, cp_len:]

    # FFT: time domain -> frequency domain, recovering the per-subcarrier
    # symbols we started with.
    freq_blocks = np.fft.fft(time_blocks, axis=1)

    return freq_blocks.flatten()


# ==========================================
# MAIN: build a small example and verify perfect reconstruction
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)

    N_SUBCARRIERS = 64
    CP_LEN = 16          # standard-ish ratio: CP = 1/4 of the block, like LTE
    N_OFDM_SYMBOLS = 20  # how many OFDM symbols to send

    n_bits = N_SUBCARRIERS * N_OFDM_SYMBOLS * 2  # 2 bits per QPSK symbol
    tx_bits = np.random.randint(0, 2, n_bits)

    print(f"Generated {n_bits} random bits.")

    # --- Stage 1: modulate to QPSK ---
    tx_symbols = bits_to_qpsk(tx_bits)
    print(f"Mapped to {len(tx_symbols)} QPSK symbols.")

    # --- Stage 2: build the OFDM time-domain signal ---
    tx_signal = ofdm_modulate(tx_symbols, N_SUBCARRIERS, CP_LEN)
    expected_len = N_OFDM_SYMBOLS * (N_SUBCARRIERS + CP_LEN)
    print(f"OFDM time-domain signal length: {len(tx_signal)} "
          f"(expected {expected_len})")
    assert len(tx_signal) == expected_len

    # --- Perfect-reconstruction test (no channel, no noise yet) ---
    # Since there's no channel or noise at this stage, demodulating what
    # we just modulated should recover the EXACT original symbols
    # (up to floating-point precision), and decoding those symbols should
    # recover the exact original bits. This confirms Stage 1 + 2 are
    # correctly implemented before we introduce any channel effects.
    rx_symbols = ofdm_demodulate(tx_signal, N_SUBCARRIERS, CP_LEN)
    max_error = np.max(np.abs(rx_symbols - tx_symbols))
    print(f"\nMax symbol reconstruction error (should be ~0, "
          f"floating point noise only): {max_error:.2e}")

    rx_bits = qpsk_to_bits(rx_symbols)
    bit_errors = np.sum(rx_bits != tx_bits)
    print(f"Bit errors after perfect-channel round trip: {bit_errors} "
          f"/ {n_bits} (should be exactly 0)")

    if bit_errors == 0 and max_error < 1e-9:
        print("\n✅ PASS: Stage 1 + 2 perfect reconstruction confirmed.")
    else:
        print("\n❌ FAIL: something is wrong -- check the implementation "
              "before moving to Stage 3.")

    # --- Plot: constellation diagram ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    axes[0].scatter(tx_symbols.real, tx_symbols.imag, alpha=0.6, label='TX')
    axes[0].set_title('Transmitted QPSK constellation')
    axes[0].set_xlabel('In-phase (I)')
    axes[0].set_ylabel('Quadrature (Q)')
    axes[0].axhline(0, color='gray', linewidth=0.5)
    axes[0].axvline(0, color='gray', linewidth=0.5)
    axes[0].set_xlim(-1.5, 1.5)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].grid(alpha=0.3)
    axes[0].set_aspect('equal')

    axes[1].scatter(rx_symbols.real, rx_symbols.imag, alpha=0.6,
                      color='green', label='RX (perfect channel)')
    axes[1].set_title('Received constellation (no channel yet)')
    axes[1].set_xlabel('In-phase (I)')
    axes[1].set_ylabel('Quadrature (Q)')
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].axvline(0, color='gray', linewidth=0.5)
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].grid(alpha=0.3)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('stage1_2_constellation.png', dpi=150, bbox_inches='tight')
    print("\nSaved plot: stage1_2_constellation.png")
