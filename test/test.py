# SPDX-FileCopyrightText: © 2026 Marian Topor (TG Baumont s.r.o.)
# SPDX-License-Identifier: Apache-2.0
"""
TinyTapeout MAC Unit — cocotb test suite

Tests the INT8 signed multiply-accumulate unit:
  acc += weight * activation
"""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge


async def reset(dut):
    """Apply reset"""
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def load_weight(dut, weight):
    """Phase 1: latch weight into weight_reg"""
    dut.ui_in.value = weight & 0xFF
    dut.uio_in.value = 0b00000001  # load_weight bit
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 1)


async def mac_compute(dut, activation):
    """Phase 2: rising edge of compute → acc += weight * activation"""
    dut.ui_in.value = activation & 0xFF
    dut.uio_in.value = 0b00000010  # compute bit (rising edge)
    await ClockCycles(dut.clk, 2)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 2)


async def clear_acc(dut):
    """Clear accumulator to zero"""
    dut.uio_in.value = 0b00000100  # clear bit
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 1)


async def read_acc16(dut):
    """Read 16-bit accumulator (LSB and MSB byte)"""
    # Read low byte
    dut.uio_in.value = 0b00000000  # byte_sel = 0
    await ClockCycles(dut.clk, 1)
    low = int(dut.uo_out.value)
    # Read high byte
    dut.uio_in.value = 0b00001000  # byte_sel = 1
    await ClockCycles(dut.clk, 1)
    high = int(dut.uo_out.value)
    # Combine, treat as signed 16-bit
    raw = (high << 8) | low
    if raw & 0x8000:
        raw -= 0x10000
    return raw


@cocotb.test()
async def test_basic_mac(dut):
    """Test 1: Basic MAC — 3 × 4 = 12"""
    dut._log.info("=== Test 1: Basic MAC ===")
    clock = Clock(dut.clk, 20, units="ns")  # 50 MHz
    cocotb.start_soon(clock.start())

    await reset(dut)
    await clear_acc(dut)

    await load_weight(dut, 3)
    await mac_compute(dut, 4)

    result = await read_acc16(dut)
    dut._log.info(f"3 * 4 = {result}")
    assert result == 12, f"Expected 12, got {result}"


@cocotb.test()
async def test_accumulate(dut):
    """Test 2: Accumulate 3*4 + 5*6 = 12 + 30 = 42"""
    dut._log.info("=== Test 2: Accumulate ===")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset(dut)
    await clear_acc(dut)

    await load_weight(dut, 3)
    await mac_compute(dut, 4)
    await load_weight(dut, 5)
    await mac_compute(dut, 6)

    result = await read_acc16(dut)
    dut._log.info(f"3*4 + 5*6 = {result}")
    assert result == 42, f"Expected 42, got {result}"


@cocotb.test()
async def test_signed(dut):
    """Test 3: Signed (-3) * 4 = -12"""
    dut._log.info("=== Test 3: Signed multiplication ===")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset(dut)
    await clear_acc(dut)

    # -3 in INT8 = 0xFD = 253
    await load_weight(dut, 0xFD)
    await mac_compute(dut, 4)

    result = await read_acc16(dut)
    dut._log.info(f"(-3) * 4 = {result}")
    assert result == -12, f"Expected -12, got {result}"


@cocotb.test()
async def test_dot_product(dut):
    """Test 4: Dot product [1,2,3,4] · [5,6,7,8] = 70"""
    dut._log.info("=== Test 4: Dot product ===")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset(dut)
    await clear_acc(dut)

    weights = [1, 2, 3, 4]
    activations = [5, 6, 7, 8]
    expected = sum(w * a for w, a in zip(weights, activations))

    for w, a in zip(weights, activations):
        await load_weight(dut, w)
        await mac_compute(dut, a)

    result = await read_acc16(dut)
    dut._log.info(f"[1,2,3,4]·[5,6,7,8] = {result}")
    assert result == expected, f"Expected {expected}, got {result}"


@cocotb.test()
async def test_max_int8(dut):
    """Test 5: Maximum INT8 multiplication 127 * 127 = 16129"""
    dut._log.info("=== Test 5: Max INT8 ===")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset(dut)
    await clear_acc(dut)

    await load_weight(dut, 127)
    await mac_compute(dut, 127)

    result = await read_acc16(dut)
    dut._log.info(f"127 * 127 = {result}")
    assert result == 16129, f"Expected 16129, got {result}"


@cocotb.test()
async def test_clear(dut):
    """Test 6: Clear sets accumulator to 0"""
    dut._log.info("=== Test 6: Clear ===")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset(dut)
    await clear_acc(dut)

    # Build up some value
    await load_weight(dut, 10)
    await mac_compute(dut, 10)

    # Clear
    await clear_acc(dut)

    result = await read_acc16(dut)
    dut._log.info(f"After clear: {result}")
    assert result == 0, f"Expected 0 after clear, got {result}"


@cocotb.test()
async def test_negative_negative(dut):
    """Test 7: Negative × Negative = Positive — (-3) × (-4) = 12

    Closes gap: previous tests covered pos×pos and neg×pos but not neg×neg.
    Validates correct two's-complement signed multiplication on both operands.
    """
    dut._log.info("=== Test 7: Negative × Negative ===")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset(dut)
    await clear_acc(dut)

    # -3 = 0xFD, -4 = 0xFC in two's complement INT8
    await load_weight(dut, 0xFD)  # -3
    await mac_compute(dut, 0xFC)  # -4

    result = await read_acc16(dut)
    dut._log.info(f"(-3) × (-4) = {result}")
    assert result == 12, f"Expected 12, got {result}"


@cocotb.test()
async def test_mixed_signed_accumulate(dut):
    """Test 8: Mixed-sign accumulation
       5×5 + (-3)×4 + 2×(-1) + (-1)×(-1) = 25 - 12 - 2 + 1 = 12

    Validates accumulator handles signed addition with mixed-sign products.
    """
    dut._log.info("=== Test 8: Mixed signed accumulate ===")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    await reset(dut)
    await clear_acc(dut)

    # (weight, activation) as unsigned bytes (two's-complement encoded)
    pairs = [
        (5, 5),         # +25
        (0xFD, 4),      # -3 × 4 = -12
        (2, 0xFF),      # 2 × -1 = -2
        (0xFF, 0xFF),   # -1 × -1 = +1
    ]
    expected = 25 + (-12) + (-2) + 1  # = 12

    for w, a in pairs:
        await load_weight(dut, w)
        await mac_compute(dut, a)

    result = await read_acc16(dut)
    dut._log.info(f"Mixed-signed accumulate: {result} (expected {expected})")
    assert result == expected, f"Expected {expected}, got {result}"


@cocotb.test()
async def test_random_stress(dut):
    """Test 9: 50 random single-MAC operations vs Python reference.

    Each iteration:
      - clear accumulator
      - load random weight ∈ [-127, 127]
      - compute on random activation ∈ [-127, 127]
      - verify result == Python's signed multiply

    Catches edge cases scripted tests miss: sign-bit boundaries (±1, ±127, 0),
    asymmetric operands, sequential weight/activation correctness.
    Reproducible via fixed seed.
    """
    dut._log.info("=== Test 9: Random stress (50 single MAC ops) ===")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())
    random.seed(42)  # reproducible

    await reset(dut)

    failures = []
    for i in range(50):
        w = random.randint(-127, 127)
        a = random.randint(-127, 127)
        expected = w * a  # max ±16129, fits in INT16

        await clear_acc(dut)
        await load_weight(dut, w & 0xFF)
        await mac_compute(dut, a & 0xFF)

        result = await read_acc16(dut)
        if result != expected:
            failures.append((i, w, a, expected, result))

    if failures:
        for f in failures[:5]:
            dut._log.error(
                f"  Iter {f[0]}: {f[1]:+4d} × {f[2]:+4d} expected {f[3]:+6d}, got {f[4]:+6d}"
            )
        assert False, f"{len(failures)}/50 random MACs failed (showing first 5)"

    dut._log.info("✓ All 50 random MACs matched Python reference")


@cocotb.test()
async def test_reset_recovery(dut):
    """Test 10: Reset mid-state cleans accumulator and unit recovers.

    Sequence:
      1. Build up state (acc = 50 × 30 = 1500)
      2. Apply reset (no clear_acc — reset alone should zero)
      3. Verify accumulator == 0
      4. Run fresh MAC (7 × 8 = 56) without re-clearing
      5. Verify result correct → unit fully recovered
    """
    dut._log.info("=== Test 10: Reset recovery ===")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Establish known state
    await reset(dut)
    await clear_acc(dut)
    await load_weight(dut, 50)
    await mac_compute(dut, 30)

    pre = await read_acc16(dut)
    assert pre == 1500, f"Pre-reset: expected 1500, got {pre}"
    dut._log.info(f"Pre-reset acc = {pre}")

    # Reset mid-flight (no explicit clear)
    await reset(dut)

    # Reset alone must zero the accumulator
    post = await read_acc16(dut)
    dut._log.info(f"Post-reset acc = {post}")
    assert post == 0, f"Reset should zero accumulator, got {post}"

    # Verify unit fully functional after reset (no clear_acc needed)
    await load_weight(dut, 7)
    await mac_compute(dut, 8)
    after = await read_acc16(dut)
    dut._log.info(f"Post-reset 7 × 8 = {after}")
    assert after == 56, f"Post-reset MAC failed: expected 56, got {after}"
