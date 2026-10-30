# Live browser demo

The deployed demo is https://landsat-cloud-lab-qam.snowywren64.chatgpt.site

It was initially published with owner-only access. The GitHub repository is
public; that does not make this separate hosted Site public.

The page runs the actual `context_residual_v2` checkpoint using ONNX Runtime Web
1.31.1 and WebAssembly in the visitor's browser. It includes nine validation
patches, RGB/NIR views, a reconstruction-mask overlay, a later reference view,
per-patch proxy MAE, and PNG downloads. It is not a satellite feed or an arbitrary
image upload service. No reference image is passed into the model's input tensor.
No server GPU or running Mac is required for the hosted demo.

Model quality is unchanged by deployment. Smooth generated regions remain
estimates. The eight-day temporal gap and single-scene validation limits still
apply. Display stretching is for visualization; model inputs remain reflectance.

## Build locally (optional)

From the repository root with its virtual environment active:

```bash
python -m pip install onnx onnxscript
python web/build_demo.py
python -m http.server 8080 --directory web/dist
```

Then open http://localhost:8080. Use an HTTP server; opening `index.html` directly
as a local file prevents normal model and sample fetching in browsers.

`export_demo.py` exports the model and prepares fixed validation examples. The
first input file also has a Python prediction exported as a numerical test
fixture. The app never loads that fixture; it calls `session.run()` for each
reconstruction. Production can omit `verification-expected.bin` after parity QA.

The original page had a runtime-loader mismatch with the general ORT JavaScript
bundle. The built demo deliberately uses the WASM-only bundle and matching
non-JSEP `.mjs`/`.wasm` files. Do not mix versions or replace only one asset.

## Verified behavior

- Real browser inference completed on two different patches.
- ONNX/WASM output versus Python: maximum absolute difference
  0.0000036061 across 65,536 reflectance values on the first patch.
- Channel view switching, mask overlay, and later-reference inspection worked.
- Measured browser inference took 129 ms and 78 ms after model/runtime loading
  in the test environment. These are not Mac benchmarks or full-scene timings.
- Initial download includes about 17 MB of model and WASM assets, plus selected
  patch data. Runtime speed and compatibility depend on the visitor's browser.

The hosted Site source is tracked separately under Sites project ID
`appgprj_6ae4d9441e688191945b8d0e617fe9b1`. Its `.openai/hosting.json` and deployment
audience belong to that source checkout. No hosting credentials are stored here.
