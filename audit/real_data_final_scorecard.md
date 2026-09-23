# Real-Data Stage Final Scorecard

Passing threshold: **90/100**.

| Review | Initial | Rework | Final | Result |
|---|---:|---|---:|---|
| 1. Authenticity & reproducibility | 74 | real static GeoJSON, provenance, COG/ARCO downloaders, download audit | 92 | PASS |
| 2. Data correctness & leakage | 86 | raw-hour split before windowing + dedicated tests | 97 | PASS |
| 3. E2E model/runtime | — | fixed/hard/adaptive routes + Lightning checkpoint/test | 96 | PASS |
| 4. Portability & packaging | 82 | relative manifests + clean-directory rebuild/retest | 98 | PASS |

## Overall: **96/100 — PASS**

### What is actually certified

- A small **real Beijing dynamic observation fixture** enters the V6 contract.
- A small **real Beijing static geospatial fixture** is rasterized and enters the model.
- Provenance/checksums are explicit.
- Train/val/test raw hours are disjoint.
- No network failure silently becomes synthetic data.
- R² forward/backward, STOP/ZOOM/adaptive routing and Lightning fit/checkpoint/test all execute.
- Production downloaders exist for UCI, Google ARCO ERA5 and ESA WorldCover COG/WMS.
- The packaged project was rebuilt and retested from a different root directory using portable relative manifests.

### What is not certified

- The bundled fixture is not ERA5/HRCLDAS/SMBFD grid truth.
- The boundary fixture is not urban morphology.
- Dynamic and static fixtures are not scientifically co-located.
- Current sandbox cannot certify external binary downloads or 4090D CUDA peak VRAM.

These are explicit next-stage dependencies rather than hidden claims.
