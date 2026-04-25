# MAC Unit — AI Inference Accelerator

## How it works

This is a **Multiply-Accumulate (MAC) unit** that performs the fundamental
operation of neural network inference: `accumulator += weight × activation`
using signed 8-bit integers (INT8).

This is the same operation that happens billions of times per second inside
Google TPUs, NVIDIA Tensor Cores, and Apple Neural Engines. This chip does
it on your desk, in real silicon you can hold in your hand.

The design uses a **2-phase protocol** to operate within the 8-bit data bus
limit of the TinyTapeout pin budget:

**Phase 1 — Load weight:**
1. Set `ui_in[7:0]` to the INT8 weight value
2. Assert `uio_in[0]` (load_weight)
3. On the next clock edge, the weight is latched into `weight_reg`

**Phase 2 — Compute:**
1. Set `ui_in[7:0]` to the INT8 activation value
2. Assert `uio_in[1]` (compute) — rising edge triggers the MAC
3. On the next clock edge, `accumulator += weight × activation`
4. The 16-bit signed accumulator can hold up to ~256 such products
   without overflow

**Read result:**
- `uio_in[3]` selects which byte of the 16-bit accumulator to output:
  - `byte_sel = 0` → `uo_out = accumulator[7:0]` (low byte)
  - `byte_sel = 1` → `uo_out = accumulator[15:8]` (high byte)

**Status outputs:**
- `uio_out[4]` — overflow flag (set if accumulator saturates)
- `uio_out[5]` — done flag (high after compute completes)
- `uio_out[6]` — sign bit (high if accumulator is negative)

**Clear:**
- Assert `uio_in[2]` (clear) → next clock zeros the accumulator

## How to test

For a simple dot product `[w₁,w₂,w₃] · [a₁,a₂,a₃]`:

```python
# Pseudocode using TinyTapeout Commander / RP2040
def mac_dot(weights, activations):
    tt.uio_in.value = 0b0100        # clear
    tt.clock_project_once()
    tt.uio_in.value = 0

    for w, a in zip(weights, activations):
        # Load weight
        tt.ui_in.value = w & 0xFF
        tt.uio_in.value = 0b0001    # load_weight
        tt.clock_project_once()
        tt.uio_in.value = 0
        # Compute
        tt.ui_in.value = a & 0xFF
        tt.uio_in.value = 0b0010    # compute
        tt.clock_project_once()
        tt.uio_in.value = 0

    # Read 16-bit result
    tt.uio_in.value = 0b0000        # byte_sel = 0
    low = tt.uo_out.value
    tt.uio_in.value = 0b1000        # byte_sel = 1
    high = tt.uo_out.value
    return (high << 8) | low

# Example: dot([1,2,3,4], [5,6,7,8]) should return 70
```

The cocotb test suite covers:
- Basic multiplication (3 × 4 = 12)
- Accumulation (3·4 + 5·6 = 42)
- Signed multiplication ((-3) × 4 = -12)
- 4-element dot product = 70
- Maximum INT8 saturation test (127 × 127 = 16129)
- Clear behavior

All 6 tests pass on the RTL simulation.

## External hardware

This is a self-contained design — no external hardware required.

You can drive it manually from the TinyTapeout Commander board (RP2040
with switches and 7-segment display), or from a microcontroller via the
PMOD-style breakout pins.

For a real-world demo, connect:
- 7-segment display to `uo[7:0]` (shows accumulator low byte)
- Two 8-position DIP switches to `ui[7:0]` (data) and `uio[3:0]` (control)
- LEDs to `uio[4:6]` (overflow, done, sign indicators)

## Why it matters

Modern AI accelerators contain thousands of these MAC units running in
parallel. This single-MAC chip is a teaching/proof-of-concept design that
demonstrates:

1. **The fundamental compute primitive** of all neural networks
2. **Real silicon implementation** on SkyWater 130nm — not just simulation
3. **A first step** toward larger custom accelerators (this MAC is a building
   block for systolic arrays, attention layers, etc.)

This is part of a longer roadmap toward open-source AI inference silicon
that anyone can fabricate, study, and improve.
