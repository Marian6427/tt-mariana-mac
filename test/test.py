# SPDX-FileCopyrightText: © 2026 Marian Topor (TG Baumont s.r.o.)
# SPDX-License-Identifier: Apache-2.0
"""
TinyTapeout MAC Unit — cocotb test suite

Tests the INT8 signed multiply-accumulate unit:
  acc += weight * activation
"""

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
