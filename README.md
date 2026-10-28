# Cat-suppression demo (runs in the browser)

Static page for GitHub Pages. Loads a 4-bit ONNX export of Gemma-2-2B-it (~2.6 GB, cached by the
browser after the first visit) and runs it on WebGPU, falling back to WASM. The ONNX graph carries
the interventions from the cat study as extra inputs: a per-layer direction to project out, and one
SAE feature per clamp slot at layers 8 and 14 to remove and push negative. Zero inputs reproduce the
stock model, so one file serves the baseline and the intervened column.

Files
- `index.html`, `app.js`: page and generation loop (onnxruntime-web, greedy decoding, KV cache).
- `interventions.json`: cat direction and the two cat features (258 KB), written by `scripts/demo_export_tensors.py`.
- `coi-serviceworker.js`: adds COOP/COEP headers on GitHub Pages so multithreaded WASM works.
- Model files are hosted on the Hugging Face Hub (see `MODEL_BASE` in `app.js`); override with `?model=<url>/`.

Build (see `scripts/`): `optimum-cli export onnx` -> `onnx_surgery.py` -> `onnx_quantize.py` (q4 weights,
fp16 activations) -> `onnx_test.py` (CPU check) -> `demo_serve_test.py` (headless browser check).
