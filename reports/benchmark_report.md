# Benchmark Report

## Configuration

- Source mode: bundled offline corpus
- Cost: provider-reported cost when available; otherwise left blank
- Quality: lightweight 0-10 heuristic from output completeness, sources, notes, citations, and errors

## Results

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline-offline | 0.05 |  | 6.0 | 0% | 0% | sources=5, tokens=96 |
| multi-agent-offline | 0.07 |  | 8.0 | 0% | 0% | sources=5, tokens=342, route=researcher->analyst->writer->done |

## Analysis

- Fastest run: `baseline-offline` at 0.05s.
- Highest heuristic quality: `multi-agent-offline` (8.0/10).
- Successful runs: 2/2.

## Limitations

- Offline fallback responses are deterministic and useful for plumbing checks, not final human-quality evaluation.
- Citation coverage only checks whether source identifiers appear in the final answer; it does not prove entailment.
- Run a real model and peer rubric before using this report as submission evidence.
