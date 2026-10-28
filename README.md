# Making Gemma forget cats, in your browser

Live: https://alexjhuang.github.io/gemma-cat-steer/

Gemma-2-2B-it runs on your own device through onnxruntime-web: WebGPU when the browser has it,
WASM on the CPU otherwise. Nothing you type leaves the machine. Pick an intervention from the
cat-suppression study and compare it with the unmodified model side by side.

What you need
- A desktop browser with WebGPU (Chrome or Edge 113+, Safari 26+, Firefox 141+ on Windows) for the
  fast path, or any modern browser for the CPU path.
- About 2.2 GB of download on the first visit (cached afterwards when the browser allows it) and
  roughly 4 GB of free RAM. Phones are not supported.

How it works
The model is a 4-bit ONNX export with extra inputs compiled into every decoder layer: a direction to
project out of the residual stream, and at layers 8 and 14 one Gemma Scope sparse-autoencoder
feature to remove and push negative. Zero inputs reproduce the stock model. The features and the
direction come from contrasting cat sentences with matched dog sentences; the study, numbers and
caveats are in https://github.com/alexjhuang/qwen-steer/blob/suppress-cats/GEMMA_REPORT.md.

Model files: https://github.com/alexjhuang/gemma-cat-steer-weights (Gemma Terms of Use apply).
