/**
 * recompiler-ui.js — boots the Xenon Recompiler page.
 *
 * Pipeline: image bytes (file or built-in sample) → parse (XEX2/ELF/bin) →
 * buildImage → discoverFragments → lift to wasm (recompileImage) → instantiate
 * with xenorecomp host imports → run on demand, print MMIO console output.
 * The same wasm + image segments feed the downloadable project zip.
 */
import { BUILTIN_PROGRAMS } from './programs.js';
import { assemble } from './ppc_asm.js';
import { classifyMagic, parseXex2, parseElf } from './inspect.js';
import {
  buildImage, recompileImage, makeImports, RecompExit, RecompUnimpl, GUEST_MEM_MASK,
} from './recomp/recompiler.js';
import { makeZip } from './recomp/zip.js';
import { buildProjectFiles } from './recomp/templates.js';

const els = {
  dropzone: document.getElementById('dropzone'),
  filePicker: document.getElementById('file-picker'),
  imageInfo: document.getElementById('image-info'),
  btnRecompile: document.getElementById('btn-recompile'),
  btnDownload: document.getElementById('btn-download'),
  statsStrip: document.getElementById('stats-strip'),
  consoleOut: document.getElementById('console-out'),
  errorBanner: document.getElementById('error-banner'),
};

const CONSOLE_CAP = 256 * 1024;

const state = {
  name: null,
  kind: null,
  built: null,     // { arena, entry, base, segments }
  wasm: null,
  stats: null,
  compileMs: 0,
};

function setInfo(text) { els.imageInfo.textContent = text; }
function showError(message) {
  els.errorBanner.textContent = message;
  els.errorBanner.style.display = message ? 'block' : 'none';
}
function setConsole(text, { hint = false } = {}) {
  els.consoleOut.textContent = text;
  els.consoleOut.classList.toggle('hint', hint);
}
function appendConsole(text) {
  if (els.consoleOut.classList.contains('hint')) setConsole('');
  const room = CONSOLE_CAP - els.consoleOut.textContent.length;
  if (room > 0) els.consoleOut.textContent += text.slice(0, room);
  els.consoleOut.scrollTop = els.consoleOut.scrollHeight;
}
function fmtBytes(n) {
  if (n >= 1024 * 1024) return (n / 1024 / 1024).toFixed(2) + ' MB';
  if (n >= 1024) return (n / 1024).toFixed(1) + ' KB';
  return n + ' B';
}
function fmtAddr(n) { return '0x' + (Number(n) >>> 0).toString(16); }
function sanitizeName(name) {
  const base = String(name).replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '');
  return base || 'project';
}

function resetOutput() {
  els.statsStrip.hidden = true;
  showError(null);
  setConsole('Pick a binary above, then recompile — the guest runs in-page via WebAssembly. Console (MMIO) output appears here.', { hint: true });
}

function setImage({ name, kind, built, extra }) {
  state.name = sanitizeName(name);
  state.kind = kind;
  state.built = built;
  state.wasm = null;
  state.stats = null;
  setInfo(`${state.name} — ${kind.toUpperCase()} · entry ${fmtAddr(built.entry)} · image ${fmtBytes(built.segments.reduce((n, s) => n + s.data.length, 0))}${extra ? ' · ' + extra : ''}`);
  els.btnRecompile.disabled = false;
  els.btnDownload.disabled = true;
  resetOutput();
}

function loadBytes(name, kind, bytes, parsed) {
  const built = buildImage(kind, parsed, bytes); // throws with a clear message on bad/encrypted images
  let extra = null;
  if (kind === 'xex2') {
    if (built.peError) extra = `PE parse failed (${built.peError}) — discovery is control-flow only`;
    else if (built.peSections) {
      const seeded = (built.functionStarts ?? []).length;
      const switches = built.switchTables?.size ?? 0;
      const vmx = built.vmxStubs?.size ?? 0;
      extra = `${built.peSections.length} PE sections · ${seeded} seeded function${seeded === 1 ? '' : 's'}${switches ? ` · ${switches} switch table${switches === 1 ? '' : 's'}` : ''}${vmx ? ` · ${vmx} VMX stub${vmx === 1 ? '' : 's'}` : ''}`;
    }
  }
  setImage({ name, kind: kind === 'xex2' ? 'xex' : kind, built, extra });
}

async function loadFile(file) {
  showError(null);
  setInfo('reading ' + file.name + '…');
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const magic = classifyMagic(bytes);
    if (magic === 'xex') {
      const parsed = parseXex2(bytes.length > 4 * 1024 * 1024 ? bytes.subarray(0, 4 * 1024 * 1024) : bytes);
      loadBytes(file.name, 'xex2', bytes, parsed);
    } else if (magic === 'elf') {
      const parsed = parseElf(bytes);
      loadBytes(file.name, 'elf', bytes, parsed);
    } else {
      loadBytes(file.name, 'bin', bytes, null);
      setInfo(els.imageInfo.textContent + ' · raw binary assumed base 0x10000');
    }
  } catch (e) {
    setImageIdle();
    showError('Could not load "' + file.name + '":\n' + (e && e.message ? e.message : String(e)));
    setInfo('');
  }
}

function setImageIdle() {
  state.name = null;
  state.kind = null;
  state.built = null;
  state.wasm = null;
  state.stats = null;
  els.btnRecompile.disabled = true;
  els.btnDownload.disabled = true;
}

function loadSample(key) {
  const needle = key.toLowerCase();
  const sample = BUILTIN_PROGRAMS.find((p) => p.name.toLowerCase().includes(needle)) ?? BUILTIN_PROGRAMS[0];
  const { bytes } = assemble(sample.source);
  const built = buildImage('bin', null, bytes);
  setImage({ name: key, kind: 'bin', built });
}

async function recompileAndRun() {
  if (!state.built) return;
  els.btnRecompile.disabled = true;
  showError(null);
  setConsole('');

  // --- compile (fresh, so the zip holds exactly this artifact) ---
  let compiled;
  const t0 = performance.now();
  try {
    compiled = recompileImage(state.built.arena, state.built.entry, { hleCalls: state.built.hleCalls, functionStarts: state.built.functionStarts, switchTables: state.built.switchTables, vmxStubs: state.built.vmxStubs });
  } catch (e) {
    els.btnRecompile.disabled = false;
    showError('Recompilation failed:\n' + (e && e.message ? e.message : String(e)));
    return;
  }
  const compileMs = performance.now() - t0;
  state.wasm = compiled.wasm;
  state.stats = compiled.stats;
  state.compileMs = compileMs;

  // --- run ---
  const entryName = 'fn_' + (state.built.entry & GUEST_MEM_MASK).toString(16);
  let exitCode = null;
  let trap = null;
  const imports = makeImports({
    onConsoleByte: (b) => appendConsole(String.fromCharCode(b)),
  });
  try {
    if (state.built.hleCalls) imports.__setHleCalls(state.built.hleCalls);
    const { instance } = await WebAssembly.instantiate(compiled.wasm, imports);
    imports.__exports = instance.exports;
    imports.__attachMemory(instance.exports.memory);
    new Uint8Array(instance.exports.memory.buffer).set(state.built.arena);
    // --- GPU subsystem: expose the ring-buffer doorbell to the HLE (Vd* exports) ---
    try {
      if (typeof globalThis.XenonGpuHostFactory === 'function' && !globalThis.XenonGpuHost) {
        globalThis.XenonGpuHost = new globalThis.XenonGpuHostFactory({
          memory: instance.exports.memory,
          onLog: (level, msg) => appendConsole(`[gpu] ${msg}\n`),
        });
      }
    } catch (e) {
      appendConsole(`[gpu] GPU subsystem unavailable: ${e && e.message ? e.message : String(e)}\n`);
    }
    const t1 = performance.now();
    try {
      instance.exports[entryName]();
      exitCode = 0;
    } catch (e) {
      if (e instanceof RecompExit) exitCode = e.code;
      else if (e instanceof RecompUnimpl) trap = e.message;
      else trap = 'trap: ' + (e && e.message ? e.message : String(e));
    }
    state.runMs = performance.now() - t1;
  } catch (e) {
    trap = 'wasm failed to instantiate: ' + (e && e.message ? e.message : String(e));
  }

  // --- report ---
  const s = state.stats;
  els.statsStrip.innerHTML = '';
  const bits = [
    ['fragments', s.fragments], ['instructions', s.insns], ['blocks', s.blocks],
    ['unimpl', s.unimpl], ['compile', compileMs.toFixed(1) + ' ms'],
    ['run', (state.runMs ?? 0).toFixed(1) + ' ms'],
    ['exit', exitCode === null ? '—' : String(exitCode)],
  ];
  for (const [k, v] of bits) {
    const span = document.createElement('span');
    span.innerHTML = `<b>${k}</b> ${v}`;
    els.statsStrip.appendChild(span);
  }
  els.statsStrip.hidden = false;

  if (trap) {
    showError(trap + '\nThis build ships a pure HLE kernel — unimplemented kernel imports trap by name (docs/HLE.md).');
  }
  if (els.consoleOut.textContent === '' && !trap) setConsole('(guest produced no console output)', { hint: true });
  els.btnDownload.disabled = false;
  els.btnRecompile.disabled = false;
}

function downloadZip() {
  if (!state.wasm || !state.built) return;
  const files = buildProjectFiles({
    name: state.name,
    kind: state.kind,
    entry: state.built.entry,
    wasm: state.wasm,
    segments: state.built.segments,
    stats: state.stats,
    hleCalls: state.built.hleCalls,
    discovery: {
      peSections: state.built.peSections?.length ?? null,
      peError: state.built.peError ?? null,
      seededFunctionStarts: state.built.functionStarts?.length ?? 0,
      switchTables: state.built.switchTables?.size ?? 0,
      vmxStubs: state.built.vmxStubs?.size ?? 0,
    },
  });
  const zip = makeZip(files);
  const blob = new Blob([zip], { type: 'application/zip' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${state.name}-recompiled.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// --- wire events ---
els.dropzone.addEventListener('click', () => els.filePicker.click());
els.dropzone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); els.filePicker.click(); } });
els.filePicker.addEventListener('change', () => { const f = els.filePicker.files?.[0]; if (f) loadFile(f); });
for (const ev of ['dragenter', 'dragover']) {
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.add('over'); });
}
for (const ev of ['dragleave', 'drop']) {
  els.dropzone.addEventListener(ev, (e) => { e.preventDefault(); els.dropzone.classList.remove('over'); });
}
els.dropzone.addEventListener('drop', (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) loadFile(f);
});
for (const chip of document.querySelectorAll('.chip[data-sample]')) {
  chip.addEventListener('click', () => loadSample(chip.dataset.sample));
}
els.btnRecompile.addEventListener('click', recompileAndRun);
els.btnDownload.addEventListener('click', downloadZip);
