import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs";
import { GemmaTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2";

// Where the model lives. Override with ?model=https://.../ to point at another host.
const params = new URLSearchParams(location.search);
const MODEL_BASE = params.get("model") || "https://huggingface.co/alexjhuang/gemma-2-2b-it-cat-steer-onnx/resolve/main/";
const $ = id => document.getElementById(id);
const st = $("status"), prog = $("prog");
let session, tok, iv, io, kvShape, stopFlag = false;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

async function fetchCached(url, label) {
  const cache = await caches.open("gemma-cat-demo-v1");
  let r = await cache.match(url);
  if (!r) {
    r = await fetch(url);
    if (!r.ok) throw new Error(`${label}: HTTP ${r.status}`);
    const total = +r.headers.get("content-length") || 0, reader = r.body.getReader(), chunks = []; let got = 0;
    prog.hidden = false;
    for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; if (total) prog.value = got / total; st.textContent = `Downloading ${label}: ${(got / 1e6).toFixed(0)} MB`; }
    const blob = new Blob(chunks);
    try { await cache.put(url, new Response(blob, { headers: { "content-length": String(blob.size) } })); }
    catch (e) { console.warn("cache put failed (quota?); continuing without cache", e); }
    return new Uint8Array(await blob.arrayBuffer());
  }
  st.textContent = `Loading ${label} from cache`;
  return new Uint8Array(await r.arrayBuffer());
}

function f16(x) {  // float32 -> IEEE half bits
  const f = new Float32Array([x]), u = new Uint32Array(f.buffer)[0], s = (u >> 16) & 0x8000; let e = ((u >> 23) & 0xff) - 112, m = u & 0x7fffff;
  if (e <= 0) { if (e < -10) return s; m = (m | 0x800000) >> (1 - e); return s | ((m + 0x1000) >> 13); }
  if (e >= 31) return s | 0x7c00;
  return s | (e << 10) | ((m + 0x1000) >> 13);
}
const toHalf = arr => { const o = new Uint16Array(arr.length); for (let i = 0; i < arr.length; i++) o[i] = f16(arr[i]); return o; };
const halfTensor = (data, dims) => new ort.Tensor("float16", toHalf(data), dims);

$("load").onclick = async () => {
  try {
    $("load").disabled = true;
    const want = $("backend").value, hasGPU = !!navigator.gpu && (await navigator.gpu.requestAdapter?.()) != null;
    const ep = want === "wasm" ? "wasm" : (want === "webgpu" || hasGPU) ? "webgpu" : "wasm";
    ort.env.wasm.proxy = false;   // the proxy worker would have to receive ~2 GB of weights via postMessage
    if (ep === "wasm") ort.env.wasm.numThreads = Math.min(8, navigator.hardwareConcurrency || 4);
    iv = await (await fetch("interventions.json")).json();
    const [tj, tc] = await Promise.all([fetch(MODEL_BASE + "tokenizer.json").then(r => r.json()), fetch(MODEL_BASE + "tokenizer_config.json").then(r => r.json())]);
    tok = new GemmaTokenizer(tj, tc);
    const manifests = await (await fetch(MODEL_BASE + "manifest.json")).json();
    const manifest = manifests[ep] ?? manifests.webgpu ?? manifests;   // q4f16 for WebGPU, q4 (fp32 activations) for WASM
    const cfg = await (await fetch(MODEL_BASE + "config.json")).json(); kvShape = [1, cfg.num_key_value_heads, 0, cfg.head_dim];
    const graph = await fetchCached(MODEL_BASE + manifest.model, "graph"); const externalData = [];
    for (const [i, name] of manifest.shards.entries()) externalData.push({ path: name, data: await fetchCached(MODEL_BASE + name, `weights ${i + 1}/${manifest.shards.length}`) });
    st.textContent = `Creating ${ep} session (compiling kernels, 10-60 s)...`;
    const opts = { executionProviders: [ep], externalData, graphOptimizationLevel: "all" };
    try { session = await ort.InferenceSession.create(graph, opts); }
    catch (e) {
      if (ep !== "webgpu" || !manifests.wasm) throw e;
      console.warn("WebGPU session failed, falling back to WASM", e); st.textContent = "WebGPU failed; loading the WASM variant...";
      ort.env.wasm.numThreads = Math.min(8, navigator.hardwareConcurrency || 4); const mw = manifests.wasm; const ext = [];
      const g2 = await fetchCached(MODEL_BASE + mw.model, "graph (wasm)"); for (const [i, name] of mw.shards.entries()) ext.push({ path: name, data: await fetchCached(MODEL_BASE + name, `weights ${i + 1}/${mw.shards.length} (wasm)`) });
      session = await ort.InferenceSession.create(g2, { executionProviders: ["wasm"], externalData: ext, graphOptimizationLevel: "all" });
    }
    io = introspect(session);
    st.innerHTML = `<span class="ok">Ready on ${session.handler?.backend?.name ?? ep}.</span> ${io.nLayers} layers, KV dtype ${io.kvType}.`; prog.hidden = true;
    $("go").disabled = false;
  } catch (e) { st.innerHTML = `<span class="warn">Load failed: ${e.message}</span>`; $("load").disabled = false; console.error(e); }
};

function introspect(s) {
  const ins = s.inputNames, outs = s.outputNames;
  const kv = ins.filter(n => n.startsWith("past_key_values")); const nLayers = kv.length / 2;
  const kvMeta = s.inputMetadata?.[kv[0]] ?? null;
  return { ins, outs, nLayers, kvType: kvMeta?.type ?? "float16", kvDims: kvMeta?.dimensions ?? kvMeta?.shape ?? null,
           hasPos: ins.includes("position_ids"), hasCachePos: ins.includes("cache_position"), f16io: (s.inputMetadata?.["ablate_dir"]?.type ?? "float16") === "float16" };
}

function interventionTensors(mode, strength) {
  const H = iv.hidden, NL = iv.layers, K = 4, C = 2;
  const dir = new Float32Array(NL * H), We = new Float32Array(C * K * H), be = new Float32Array(C * K), Wd = new Float32Array(C * H * K), neg = new Float32Array(C);
  if (mode === "ablate") for (const L of iv.direction.apply_layers) dir.set(iv.direction.v, L * H);
  if (mode === "clamp8" || mode === "clamp8_14") {
    const slots = mode === "clamp8" ? [0] : [0, 1];
    for (const c of slots) { const f = iv.features[c]; We.set(f.W_enc, c * K * H); be[c * K] = f.b_enc; for (let d = 0; d < H; d++) Wd[c * H * K + d * K] = f.W_dec[d]; neg[c] = strength; }
  }
  const mk = (a, dims) => io.f16io ? halfTensor(a, dims) : new ort.Tensor("float32", a, dims);
  return { ablate_dir: mk(dir, [NL, H]), clamp_We: mk(We, [C, K, H]), clamp_be: mk(be, [C, K]), clamp_Wd: mk(Wd, [C, H, K]), clamp_neg: mk(neg, [C]) };
}

const chat = p => `<bos><start_of_turn>user\n${p}<end_of_turn>\n<start_of_turn>model\n`;

async function* generate(prompt, ivT, maxNew) {
  const enc = tok(chat(prompt), { add_special_tokens: false });
  let ids = Array.from(enc.input_ids.data, Number); const T0 = ids.length;
  let past = {}; for (const n of io.ins.filter(n => n.startsWith("past_key_values"))) past[n] = new ort.Tensor(io.kvType, io.kvType === "float16" ? new Uint16Array(0) : new Float32Array(0), kvShape);
  let cur = ids, pos = 0, out = [], t0 = performance.now();
  for (let step = 0; step < maxNew && !stopFlag; step++) {
    const S = cur.length, total = pos + S;
    const feeds = { input_ids: new ort.Tensor("int64", BigInt64Array.from(cur.map(BigInt)), [1, S]), attention_mask: new ort.Tensor("int64", new BigInt64Array(total).fill(1n), [1, total]), ...past, ...ivT };
    if (io.hasPos) feeds.position_ids = new ort.Tensor("int64", BigInt64Array.from({ length: S }, (_, i) => BigInt(pos + i)), [1, S]);
    if (io.hasCachePos) feeds.cache_position = new ort.Tensor("int64", BigInt64Array.from({ length: S }, (_, i) => BigInt(pos + i)), [S]);
    const res = await session.run(feeds);
    const logits = res.logits; const V = logits.dims[2], off = (S - 1) * V; let best = 0, bv = -Infinity;
    const ld = logits.data; const read = logits.type === "float16" ? i => halfToFloat(ld[i]) : i => ld[i];
    for (let i = 0; i < V; i++) { const v = read(off + i); if (v > bv) { bv = v; best = i; } }
    for (const n of Object.keys(past)) { const pn = n.replace("past_key_values", "present"); past[n] = res[pn]; }
    pos = total; cur = [best]; out.push(best);
    if (best === tok.eos_token_id || best === 107 /* <end_of_turn> */ || best === 1) break;
    yield { text: tok.decode(out, { skip_special_tokens: true }), tps: out.length / ((performance.now() - t0) / 1000) };
  }
}
function halfToFloat(h) { const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff; if (e === 0) return s * m * 2 ** -24; if (e === 31) return m ? NaN : s * Infinity; return s * (1 + m / 1024) * 2 ** (e - 15); }

$("strength").oninput = e => $("strv").textContent = e.target.value;
$("stop").onclick = () => { stopFlag = true; };
$("go").onclick = async () => {
  stopFlag = false; $("go").disabled = true; $("stop").disabled = false; $("outA").textContent = $("outB").textContent = "";
  const prompt = $("prompt").value, mode = $("mode").value, strength = +$("strength").value, maxNew = +$("maxtok").value;
  $("tagB").textContent = mode === "none" ? "(no intervention selected)" : $("mode").selectedOptions[0].textContent + (mode === "ablate" ? "" : `, strength ${strength}`);
  try {
    for await (const r of generate(prompt, interventionTensors("none", 0), maxNew)) { $("outA").textContent = r.text; $("tps").textContent = `${r.tps.toFixed(1)} tok/s`; }
    if (mode !== "none") for await (const r of generate(prompt, interventionTensors(mode, strength), maxNew)) { $("outB").textContent = r.text; $("tps").textContent = `${r.tps.toFixed(1)} tok/s`; }
  } catch (e) { $("outB").textContent = "Error: " + (e?.message ?? String(e)); console.error(e); }
  $("go").disabled = false; $("stop").disabled = true;
};
