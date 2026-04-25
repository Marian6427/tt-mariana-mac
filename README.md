# tt_um_mariana_mac — INT8 MAC Unit for TinyTapeout

[![test](https://github.com/Marian6427/tt-mariana-mac/actions/workflows/test.yaml/badge.svg)](https://github.com/Marian6427/tt-mariana-mac/actions/workflows/test.yaml)
[![gds](https://github.com/Marian6427/tt-mariana-mac/actions/workflows/gds.yaml/badge.svg)](https://github.com/Marian6427/tt-mariana-mac/actions/workflows/gds.yaml)

A signed 8-bit Multiply-Accumulate (MAC) unit for AI inference, designed for
the [TinyTapeout](https://tinytapeout.com) shuttle program.

## What it does

`accumulator += weight × activation` using INT8 signed arithmetic — the
fundamental building block of every neural network.

## Submission target

- **Shuttle:** TTSKY26a (SkyWater 130nm)
- **Tile size:** 1×1 (947 gates synthesized)
- **Clock:** 50 MHz
- **Top module:** `tt_um_mariana_mac`

## Quick test

```bash
cd test
make
```

Should output: `TESTS=6 PASS=6 FAIL=0`

## Author

Marian Topor (TG Baumont s.r.o., Slovakia)

## License

Apache 2.0 — see [LICENSE](LICENSE)
