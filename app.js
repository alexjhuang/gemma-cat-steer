import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs";
import { GemmaTokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2";

// Where the model lives. Override with ?model=https://.../ to point at another host.
const params = new URLSearchParams(location.search);
const MODEL_BASE = params.get("model") || "https://raw.githubusercontent.com/alexjhuang/gemma-cat-steer-weights/main/";
const $ = id => document.getElementById(id);
const st = $("status"), prog = $("prog");
let session, tok, iv, io, kvShape, stopFlag = false;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
if (params.get("debug")) { ort.env.logLevel = "verbose"; ort.env.debug = true; }

async function fetchFile(manifests, name, label) {
  const entry = manifests.files?.[name], hashes = manifests.hashes ?? {};
  if (!entry || entry.chunks.length === 1 && entry.chunks[0] === name) return fetchCached(MODEL_BASE + name, label, hashes[name]);
  const out = new Uint8Array(entry.size); let off = 0;
  for (const [i, c] of entry.chunks.entries()) { const part = await fetchCached(MODEL_BASE + c, `${label} part ${i + 1}/${entry.chunks.length}`, hashes[c]); out.set(part, off); off += part.length; }
  return out;
}

const sha256hex = async buf => [...new Uint8Array(await crypto.subtle.digest("SHA-256", buf))].map(b => b.toString(16).padStart(2, "0")).join("");
let verified = 0;

// Returns the bytes for `url`. If `expected` (sha256 hex from the manifest) is given, a cached copy
// whose hash does not match is evicted and the file is refetched past the CDN cache; a network copy
// that does not match is an error (the CDN is still serving an old version; retry later).
async function fetchCached(url, label, expected) {
  const cache = await caches.open("gemma-cat-demo-v2");
  let r = await cache.match(url);
  if (r && expected) {
    const bytes = new Uint8Array(await r.arrayBuffer());
    if (await sha256hex(bytes) === expected) { verified++; st.textContent = `Loading ${label} from cache (verified)`; return bytes; }
    console.warn("stale cache entry, refetching", url); await cache.delete(url); r = null; url += (url.includes("?") ? "&" : "?") + "h=" + expected.slice(0, 12);
  }
  if (!r) {
    r = await fetch(url);
    for (let attempt = 1; !r.ok && attempt <= 3; attempt++) {   // files propagate through GitHub's CDN for a few minutes after an update
      st.textContent = `${label}: HTTP ${r.status}, retrying in ${10 * attempt} s (files may still be propagating)...`;
      await new Promise(res => setTimeout(res, 10000 * attempt));
      r = await fetch(url + (url.includes("?") ? "&" : "?") + "r=" + Date.now());
    }
    if (!r.ok) throw new Error(`${label}: HTTP ${r.status} after retries. The files are probably still propagating; try again in a few minutes.`);
    const total = +r.headers.get("content-length") || 0, reader = r.body.getReader(), chunks = []; let got = 0;
    prog.hidden = false;
    for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); got += value.length; if (total) prog.value = got / total; st.textContent = `Downloading ${label}: ${(got / 1e6).toFixed(0)} MB`; }
    const blob = new Blob(chunks); const bytes = new Uint8Array(await blob.arrayBuffer());
    if (expected) { const h = await sha256hex(bytes); if (h !== expected) throw new Error(`${label}: downloaded file does not match the manifest hash (CDN may still serve an old copy; retry in a few minutes)`); verified++; }
    try { await cache.put(url.split("?")[0], new Response(blob, { headers: { "content-length": String(blob.size) } })); }
    catch (e) { console.warn("cache put failed (quota?); continuing without cache", e); }
    return bytes;
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
    const manifests = await (await fetch(MODEL_BASE + "manifest.json?t=" + Date.now(), { cache: "no-store" })).json();   // always fresh: it carries the hashes
    const manifest = manifests[ep] ?? manifests.webgpu ?? manifests;   // q4f16 for WebGPU, q4 (fp32 activations) for WASM
    const cfg = await (await fetch(MODEL_BASE + "config.json")).json(); kvShape = [1, cfg.num_key_value_heads, 0, cfg.head_dim];
    const graph = await fetchFile(manifests, manifest.model, "graph"); const externalData = [];
    for (const [i, name] of manifest.shards.entries()) externalData.push({ path: name, data: await fetchFile(manifests, name, `weights ${i + 1}/${manifest.shards.length}`) });
    st.textContent = `Creating ${ep} session (compiling kernels, 10-60 s)...`;
    const opts = { executionProviders: [ep], externalData, graphOptimizationLevel: "all" }; let usedManifest = manifest;
    try { session = await ort.InferenceSession.create(graph, opts); }
    catch (e) {
      if (ep !== "webgpu" || !manifests.wasm) throw e;
      console.warn("WebGPU session failed, falling back to WASM", e); st.textContent = "WebGPU failed; loading the WASM variant...";
      ort.env.wasm.numThreads = Math.min(8, navigator.hardwareConcurrency || 4); const mw = manifests.wasm; const ext = [];
      const g2 = await fetchFile(manifests, mw.model, "graph (wasm)"); for (const [i, name] of mw.shards.entries()) ext.push({ path: name, data: await fetchFile(manifests, name, `weights ${i + 1}/${mw.shards.length} (wasm)`) });
      session = await ort.InferenceSession.create(g2, { executionProviders: ["wasm"], externalData: ext, graphOptimizationLevel: "all" }); usedManifest = mw;
    }
    io = introspect(session, usedManifest.dtype ?? (usedManifest.model.includes("f16") ? "float16" : "float32"));
    st.innerHTML = `<span class="ok">Ready on ${session.handler?.backend?.name ?? ep}.</span> ${io.nLayers} layers, KV dtype ${io.kvType}, ${verified} files verified by SHA-256.` + (io.hasResid ? "" : ` <span class="warn">This graph has no activation outputs (old cached copy); views are off. Clear site data and reload to get them.</span>`); prog.hidden = true;
    if (!io.hasResid) { $("viz").checked = false; $("viz").disabled = true; }
    setStep(2);
    $("go").disabled = false;
  } catch (e) { st.innerHTML = `<span class="warn">Load failed: ${e.message}</span>`; $("load").disabled = false; console.error(e); }
};

function introspect(s, dtype) {
  const ins = s.inputNames, outs = s.outputNames;
  const kv = ins.filter(n => n.startsWith("past_key_values")); const nLayers = kv.length / 2;
  return { ins, outs, nLayers, kvType: dtype, hasPos: ins.includes("position_ids"), hasCachePos: ins.includes("cache_position"), f16io: dtype === "float16", hasResid: outs.includes("resid.0") };
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
const CAT_RE = /cat|kitt|feline|meow|purr|tabby|whisker/i;
const H = 2304, NL = 26;

class Viz {
  constructor(k) { this.k = k; this.el = { strip: $("strip" + k), heat: $("heat" + k), race: $("race" + k), stack: $("stack" + k), box: $("viz" + k) }; this.reset(); }
  reset() { this.tokens = []; this.a8 = []; this.a14 = []; this.proj = []; this.isTpl = []; this.top = []; this.firstTop = null; this.el.strip.innerHTML = ""; this.el.race.innerHTML = ""; this.el.stack.innerHTML = ""; const c = this.el.heat.getContext("2d"); c.clearRect(0, 0, this.el.heat.width, this.el.heat.height); }
  // resid: array of 26 Float32Array (S*H) for the positions just processed; tokIds: ids of those positions; skipFirst: BOS at position 0
  add(tokIds, resid, top5, skipFirst, nTemplate) {
    const S = tokIds.length, f8 = iv.features[0], f14 = iv.features[1], v = iv.direction.v;
    for (let t = 0; t < S; t++) {
      const bos = skipFirst && t === 0; const tpl = this.tokens.length < nTemplate;
      let a8 = 0, a14 = 0; const pr = new Float32Array(NL);
      if (!bos) {
        const r8 = resid[8].subarray(t * H, (t + 1) * H), r14 = resid[14].subarray(t * H, (t + 1) * H);
        a8 = Math.max(0, dot(r8, f8.W_enc) + f8.b_enc); a14 = Math.max(0, dot(r14, f14.W_enc) + f14.b_enc);
        for (let L = 0; L < NL; L++) pr[L] = dot(resid[L].subarray(t * H, (t + 1) * H), v);
      }
      this.tokens.push(tok.decode([tokIds[t]])); this.a8.push(a8); this.a14.push(a14); this.proj.push(pr); this.isTpl.push(tpl || bos);
    }
    if (top5) { this.top = top5; if (!this.firstTop) this.firstTop = top5; }   // the first step decides the answer word
    this.render();
  }
  render() {
    const T = this.tokens.length; if (!T) return;
    // 1. feature strip
    this.el.strip.innerHTML = this.tokens.map((s, i) => `<span class="${this.isTpl[i] ? "tpl" : ""}" style="--a:${Math.min(1, this.a8[i] / 30).toFixed(2)};--b:${Math.min(1, this.a14[i] / 30).toFixed(2)}" title="L8 ${this.a8[i].toFixed(1)} · L14 ${this.a14[i].toFixed(1)}">${esc(s)}</span>`).join("");
    // 2. heatmap: per-layer centring on the median over tokens, scaled by the per-layer max |dev|
    const cv = this.el.heat; cv.width = Math.max(T, 1); cv.height = NL; const c = cv.getContext("2d"); const img = c.createImageData(cv.width, NL);
    for (let L = 0; L < NL; L++) {
      const vals = this.proj.map(p => p[L]).filter((_, i) => !this.isTpl[i]); const med = median(vals.length ? vals : [0]); let mx = 1e-6; for (const x of vals) mx = Math.max(mx, Math.abs(x - med));
      for (let t = 0; t < T; t++) { const z = this.isTpl[t] ? 0 : (this.proj[t][L] - med) / mx; const o = (L * cv.width + t) * 4; const r = z > 0 ? 247 : 122, g = z > 0 ? 118 : 162, b = z > 0 ? 142 : 247; const a = Math.min(1, Math.abs(z)); img.data[o] = 15 + (r - 15) * a; img.data[o + 1] = 17 + (g - 17) * a; img.data[o + 2] = 21 + (b - 21) * a; img.data[o + 3] = 255; }
    }
    c.putImageData(img, 0, 0);
    // 3. next-token race
    const bars = top => top.map(([s, p]) => `<div class="${CAT_RE.test(s) ? "cat" : ""}"><span class="t">${esc(s)}</span><b style="width:${Math.max(1, p * 100).toFixed(0)}%"></b><span>${(p * 100).toFixed(0)}%</span></div>`).join("");
    if (this.firstTop) this.el.race.innerHTML = `<div style="color:var(--mute);font-size:12px">first answer token</div>${bars(this.firstTop)}` + (this.top !== this.firstTop ? `<div style="color:var(--mute);font-size:12px;margin-top:6px">latest step</div>${bars(this.top)}` : "");
    // 4. layer stack: glow = |centred projection| at the latest non-template token
    let last = T - 1; while (last > 0 && this.isTpl[last]) last--;
    const w = 260 / NL; let svg = "";
    for (let L = 0; L < NL; L++) {
      const vals = this.proj.map(p => p[L]).filter((_, i) => !this.isTpl[i]); const med = median(vals.length ? vals : [0]); let mx = 1e-6; for (const x of vals) mx = Math.max(mx, Math.abs(x - med));
      const z = this.isTpl[last] ? 0 : (this.proj[last][L] - med) / mx; const col = z > 0 ? "247,118,142" : "122,162,247";
      const edited = this.edited?.(L); svg += `<rect x="${(L * w).toFixed(1)}" y="6" width="${(w - 1.5).toFixed(1)}" height="40" rx="2" fill="rgba(${col},${Math.min(1, Math.abs(z)).toFixed(2)})" stroke="${edited ? "#e8e8ea" : "#2a2f3b"}" stroke-width="${edited ? 1.5 : 0.5}"><title>layer ${L}: ${this.proj[last][L].toFixed(1)}</title></rect>`;
    }
    this.el.stack.innerHTML = svg + `<text x="0" y="55" fill="#8b93a7" font-size="7">layer 0</text><text x="236" y="55" fill="#8b93a7" font-size="7">25</text>`;
  }
}
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const median = a => { const b = [...a].sort((x, y) => x - y); return b[b.length >> 1]; };
const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/\n/g, "⏎");
const vizA = new Viz("A"), vizB = new Viz("B");
function toF32(t) { if (t.type !== "float16") return t.data; const d = t.data, o = new Float32Array(d.length); for (let i = 0; i < d.length; i++) o[i] = halfToFloat(d[i]); return o; }
function top5(logits, off, V) { const read = logits.type === "float16" ? i => halfToFloat(logits.data[i]) : i => logits.data[i]; let mx = -Infinity; for (let i = 0; i < V; i++) mx = Math.max(mx, read(off + i)); let sum = 0; const best = []; for (let i = 0; i < V; i++) { const e = Math.exp(read(off + i) - mx); sum += e; if (best.length < 5 || e > best[4][1]) { best.push([i, e]); best.sort((a, b) => b[1] - a[1]); if (best.length > 5) best.pop(); } } return best.map(([i, e]) => [tok.decode([i]), e / sum]); }

async function* generate(prompt, ivT, maxNew, viz) {
  const enc = tok(chat(prompt), { add_special_tokens: false });
  let ids = Array.from(enc.input_ids.data, Number); const T0 = ids.length;
  const nTemplate = tok("<bos><start_of_turn>user\n", { add_special_tokens: false }).input_ids.data.length;
  const fetches = viz ? ["logits", ...io.outs.filter(n => n.startsWith("present")), ...Array.from({ length: NL }, (_, L) => `resid.${L}`)] : null;
  if (viz) viz.reset();
  let past = {}; for (const n of io.ins.filter(n => n.startsWith("past_key_values"))) past[n] = new ort.Tensor(io.kvType, io.kvType === "float16" ? new Uint16Array(0) : new Float32Array(0), kvShape);
  let cur = ids, pos = 0, out = [], t0 = performance.now();
  for (let step = 0; step < maxNew && !stopFlag; step++) {
    const S = cur.length, total = pos + S;
    const feeds = { input_ids: new ort.Tensor("int64", BigInt64Array.from(cur.map(BigInt)), [1, S]), attention_mask: new ort.Tensor("int64", new BigInt64Array(total).fill(1n), [1, total]), ...past, ...ivT };
    if (io.hasPos) feeds.position_ids = new ort.Tensor("int64", BigInt64Array.from({ length: S }, (_, i) => BigInt(pos + i)), [1, S]);
    if (io.hasCachePos) feeds.cache_position = new ort.Tensor("int64", BigInt64Array.from({ length: S }, (_, i) => BigInt(pos + i)), [S]);
    const res = fetches ? await session.run(feeds, fetches) : await session.run(feeds);
    const logits = res.logits; const V = logits.dims[2], off = (S - 1) * V; let best = 0, bv = -Infinity;
    const ld = logits.data; const read = logits.type === "float16" ? i => halfToFloat(ld[i]) : i => ld[i];
    let bad = 0; for (let i = 0; i < V; i++) { const v = read(off + i); if (v !== v || v === Infinity || v === -Infinity) { bad++; continue; } if (v > bv) { bv = v; best = i; } }
    if (bad > V / 2 || bv === -Infinity) {
      let where = "";
      if (fetches) { for (let L = 0; L < NL; L++) { const d = toF32(res[`resid.${L}`]); let n = 0; for (let i = 0; i < d.length; i++) if (d[i] !== d[i] || d[i] === Infinity || d[i] === -Infinity) n++; if (n) { where = `; first non-finite residual at layer ${L} (${n} of ${d.length} values)`; break; } } }
      const gpu = (await navigator.gpu?.requestAdapter?.())?.info; const adapter = gpu ? ` [${gpu.vendor ?? ""} ${gpu.architecture ?? ""} ${gpu.description ?? ""}]`.replace(/\s+/g, " ") : "";
      throw new Error(`model produced non-finite logits on this backend (${bad} of ${V})${where}${adapter}; switch Backend to WASM`);
    }
    if (viz) { const resid = Array.from({ length: NL }, (_, L) => toF32(res[`resid.${L}`])); viz.add(cur, resid, top5(logits, off, V), pos === 0, nTemplate); }
    for (const n of Object.keys(past)) { const pn = n.replace("past_key_values", "present"); past[n] = res[pn]; }
    pos = total; cur = [best]; out.push(best);
    if (best === tok.eos_token_id || best === 107 /* <end_of_turn> */ || best === 1) break;
    const text = tok.decode(out, { skip_special_tokens: true });
    yield { text: text.trim() ? text : `[${out.length} token${out.length > 1 ? "s" : ""} with no visible text: ids ${out.slice(0, 8).join(",")}]`, tps: out.length / ((performance.now() - t0) / 1000) };
  }
}
function halfToFloat(h) { const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff; if (e === 0) return s * m * 2 ** -24; if (e === 31) return m ? NaN : s * Infinity; return s * (1 + m / 1024) * 2 ** (e - 15); }

$("strength").oninput = e => $("strv").textContent = e.target.value;
document.querySelectorAll("#modes .mode").forEach(m => m.onclick = () => { document.querySelectorAll("#modes .mode").forEach(x => x.classList.remove("sel")); m.classList.add("sel"); $("mode").value = m.dataset.mode; $("tagB").textContent = m.querySelector("b").textContent.toLowerCase(); setStep(session ? 3 : 1); });
document.querySelectorAll("#chips .chip").forEach(c => c.onclick = () => { $("prompt").value = c.textContent; });
function setStep(n) { for (let i = 1; i <= 3; i++) { const el = $("s" + i); el.classList.toggle("active", i === n); el.classList.toggle("done", i < n); } }
$("backend").onchange = async () => {   // the backend is fixed when the session is created: drop it and ask for a new Load
  if (!session) return;
  try { await session.release?.(); } catch {} session = null; io = null;
  $("go").disabled = true; $("load").disabled = false; setStep(1);
  st.innerHTML = `Backend changed to <b>${$("backend").selectedOptions[0].textContent}</b>. Click <b>Load model</b> to switch (the other variant downloads once, ~2.2 GB, then is cached).`;
};
$("stop").onclick = () => { stopFlag = true; };
$("go").onclick = async () => {
  stopFlag = false; $("go").disabled = true; $("stop").disabled = false; $("outA").textContent = $("outB").textContent = "";
  const prompt = $("prompt").value, mode = $("mode").value, strength = +$("strength").value, maxNew = +$("maxtok").value;
  const modeCard = document.querySelector(`#modes .mode[data-mode="${mode}"] b`)?.textContent ?? mode;
  $("tagB").textContent = mode === "none" ? "no intervention selected" : modeCard.toLowerCase() + (mode === "ablate" ? "" : ` · strength ${strength}`); setStep(3);
  try {
    const useViz = $("viz").checked; document.querySelectorAll(".viz").forEach(e => e.style.display = useViz ? "" : "none");
    vizA.edited = () => false; vizB.edited = L => mode === "ablate" ? L >= 8 : mode === "clamp8" ? L === 8 : mode === "clamp8_14" ? (L === 8 || L === 14) : false;
    for await (const r of generate(prompt, interventionTensors("none", 0), maxNew, useViz ? vizA : null)) { $("outA").textContent = r.text; $("tps").textContent = `${r.tps.toFixed(1)} tok/s`; }
    if (mode !== "none") for await (const r of generate(prompt, interventionTensors(mode, strength), maxNew, useViz ? vizB : null)) { $("outB").textContent = r.text; $("tps").textContent = `${r.tps.toFixed(1)} tok/s`; }
  } catch (e) { $("outB").textContent = "Error: " + (e?.message ?? String(e)); console.error(e); }
  $("go").disabled = false; $("stop").disabled = true;
};
