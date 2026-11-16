/**
 * main.js — application wiring for Xenia Canary Web.
 * Views: Library / Run / Inspector / Settings / About.
 * No network calls after load; no telemetry.
 */
import { GameLibrary, formatBytes, MAX_FULL_HASH_BYTES, LibraryUnavailable } from './storage.js';
import { Runner } from './runner.js';
import { BUILTIN_PROGRAMS } from './programs.js';
import { parseXex2, parseElf, classifyMagic, xexPageTypeName, ELF_PT } from './inspect.js';
import { formatHex } from './core.js';

const APP_VERSION = '0.1.0';
const SETTINGS_KEY = 'xeniatest.settings';
const DIAG_LOG_MAX = 200;

const $ = (id) => document.getElementById(id);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/* ---------------- Settings ---------------- */
const defaultSettings = {
  trapOnUnimplemented: true,
  budget: 65536,
  fullHash: true,
  perfOverlay: false,
  confirmDelete: true,
  verbose: false,
};

function loadSettings() {
  try {
    return { ...defaultSettings, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}') };
  } catch {
    return { ...defaultSettings };
  }
}
function saveSettings(s) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
}
const settings = loadSettings();

const diagnostics = {
  app: 'xeniatest',
  version: APP_VERSION,
  startedAt: new Date().toISOString(),
  userAgent: navigator.userAgent,
  crossOriginIsolated: window.crossOriginIsolated === true,
  sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
  hardwareConcurrency: navigator.hardwareConcurrency ?? null,
  deviceMemory: navigator.deviceMemory ?? null,
  webgpu: !!navigator.gpu,
  cores: navigator.hardwareConcurrency ?? 'unknown',
  log: [],
};
function diagLog(message) {
  diagnostics.log.push({ t: new Date().toISOString(), message });
  if (diagnostics.log.length > DIAG_LOG_MAX) diagnostics.log.shift();
  if (settings.verbose) console.debug('[xeniatest]', message);
}

/* ---------------- Toasts ---------------- */
function toast(message, kind = 'ok') {
  const host = $('toast-host');
  const el = document.createElement('div');
  el.className = `toast${kind === 'ok' ? '' : ` ${kind}`}`;
  el.textContent = message;
  host.appendChild(el);
  setTimeout(() => { el.remove(); }, 5200);
  diagLog(`toast(${kind}): ${message}`);
}

/* ---------------- Status bar & footer ---------------- */
function setStatusMsg(text) { $('status-msg').textContent = text; }

async function refreshStorageStatus() {
  try {
    const { used, quota } = await library.estimate();
    $('status-storage').textContent = quota ? `${formatBytes(used)} / ${formatBytes(quota)}` : 'n/a';
    $('storage-used').textContent = `${formatBytes(used)} used`;
    $('storage-quota').textContent = quota ? `of ${formatBytes(quota)} available` : 'quota unknown';
    $('storage-fill').style.width = quota ? `${Math.min(100, (used / quota) * 100).toFixed(2)}%` : '0%';
    $('storage-meter').hidden = false;
  } catch {
    $('status-storage').textContent = 'n/a';
  }
}

function refreshThreadsStatus() {
  const el = $('status-threads');
  if (window.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined') {
    el.textContent = `SAB ready · ${navigator.hardwareConcurrency ?? '?'} cores`;
  } else {
    el.textContent = 'crossOriginIsolated OFF';
  }
}

/* ---------------- Tabs ---------------- */
const TABS = ['library', 'run', 'inspector', 'settings', 'about'];
function activateTab(name) {
  for (const tab of TABS) {
    $(`tab-${tab}`).setAttribute('aria-selected', tab === name ? 'true' : 'false');
    $(`view-${tab}`).classList.toggle('view-active', tab === name);
  }
}
for (const tab of TABS) {
  $(`tab-${tab}`).addEventListener('click', () => activateTab(tab));
}

/* ---------------- Library ---------------- */
let library = null;
try {
  library = await GameLibrary.open();
  diagLog('OPFS library opened');
} catch (error) {
  if (error instanceof LibraryUnavailable) {
    toast('OPFS is not available — the dump library is disabled.', 'warn');
    diagLog(`library unavailable: ${error.message}`);
  } else {
    toast(`Failed to open the local library: ${error.message}`, 'error');
    diagLog(`library open failed: ${error.message}`);
  }
}

const KIND_LABEL = { xex: 'XEX2', iso: 'ISO', elf: 'ELF', bin: 'RAW' };
const KIND_CLASS = { xex: 'badge-xex', iso: 'badge-iso', elf: 'badge-elf', bin: 'badge-bin' };

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function renderLibrary() {
  const grid = $('library-grid');
  const empty = $('library-empty');
  grid.innerHTML = '';
  const entries = library ? libraryEntries : [];
  empty.style.display = entries.length === 0 ? 'block' : 'none';
  for (const entry of entries) {
    const card = document.createElement('div');
    card.className = 'game-card';
    const kind = KIND_LABEL[entry.kind] ?? 'BIN';
    card.innerHTML = `
      <div class="game-card-top">
        <div class="game-badge ${KIND_CLASS[entry.kind] ?? 'badge-bin'}">${kind}</div>
        <div>
          <div class="game-name" title="${escapeHtml(entry.name)}">${escapeHtml(entry.name)}</div>
          <div class="game-meta">${formatBytes(entry.size)} · imported ${new Date(entry.importedAt).toLocaleDateString()}<br />
          ${entry.sha256 ? `sha256 ${entry.sha256.slice(0, 16)}…` : entry.quickHash ? `fingerprint ${entry.quickHash.slice(0, 16)}…` : 'no fingerprint'}</div>
        </div>
      </div>
      <div class="game-actions">
        <button class="btn btn-sm btn-primary" data-action="inspect">Inspect</button>
        <button class="btn btn-sm" data-action="run">${entry.kind === 'xex' ? 'Boot…' : 'Run'}</button>
        <button class="btn btn-sm btn-ghost" data-action="delete" aria-label="Delete">✕</button>
      </div>`;
    card.querySelector('[data-action="inspect"]').addEventListener('click', () => openInspectorFor(entry.id));
    card.querySelector('[data-action="run"]').addEventListener('click', () => runLibraryEntry(entry));
    card.querySelector('[data-action="delete"]').addEventListener('click', () => deleteLibraryEntry(entry));
    grid.appendChild(card);
  }
  refreshStorageStatus();
}

function renderBuiltins() {
  const grid = $('builtin-grid');
  grid.innerHTML = '';
  for (const program of BUILTIN_PROGRAMS) {
    const card = document.createElement('div');
    card.className = 'game-card';
    card.innerHTML = `
      <div class="game-card-top">
        <div class="game-badge badge-xex">PPC</div>
        <div>
          <div class="game-name">${escapeHtml(program.name)}</div>
          <div class="game-meta">built-in · assembles at load</div>
        </div>
      </div>
      <div class="game-actions">
        <button class="btn btn-sm btn-primary" data-action="run">Run</button>
        <button class="btn btn-sm" data-action="inspect">View asm</button>
      </div>`;
    card.querySelector('[data-action="run"]').addEventListener('click', () => {
      activateTab('run');
      loadBuiltinIntoRunner(program);
    });
    card.querySelector('[data-action="inspect"]').addEventListener('click', () => {
      activateTab('inspector');
      showBuiltinSource(program);
    });
    grid.appendChild(card);
  }
}

let libraryEntries = [];

async function refreshLibrary() {
  if (library) {
    libraryEntries = await library.list();
  } else {
    libraryEntries = [];
  }
  renderLibrary();
  refreshInspectorSourceOptions();
}

async function importFiles(fileList) {
  if (!library) { toast('Library unavailable — cannot import.', 'error'); return; }
  const files = [...fileList];
  const progress = $('import-progress');
  const bar = progress.querySelector('div');
  progress.hidden = false;
  bar.style.width = '0%';
  let done = 0;
  for (const file of files) {
    try {
      setStatusMsg(`importing ${file.name}…`);
      diagLog(`import start: ${file.name} (${file.size} B)`);
      const entry = await library.importFile(file, {
        onProgress: (written, total) => {
          const frac = total ? written / total : 0;
          bar.style.width = `${Math.round(((done + frac) / files.length) * 100)}%`;
        },
      });
      diagLog(`import ok: ${entry.name} as ${entry.id}`);
      toast(`Imported ${file.name}`, 'ok');
      done++;
    } catch (error) {
      diagLog(`import failed: ${file.name}: ${error.message}`);
      toast(`Import failed for ${file.name}: ${error.message}`, 'error');
    }
  }
  bar.style.width = '100%';
  setTimeout(() => { progress.hidden = true; bar.style.width = '0%'; }, 800);
  setStatusMsg('ready');
  await refreshLibrary();
}

async function deleteLibraryEntry(entry) {
  if (settings.confirmDelete && !window.confirm(`Delete "${entry.name}" from the library? This removes the local copy.`)) return;
  try {
    await library.delete(entry.id);
    toast(`Deleted ${entry.name}`, 'ok');
    diagLog(`deleted ${entry.id}`);
    await refreshLibrary();
  } catch (error) {
    toast(`Delete failed: ${error.message}`, 'error');
  }
}

async function runLibraryEntry(entry) {
  if (entry.kind === 'xex') {
    activateTab('run');
    const banner = $('run-banner');
    banner.className = 'banner banner-warn visible';
    banner.textContent = 'Xbox 360 titles cannot boot in the web core yet: xboxkrnl (the kernel), Xenos GPU and AltiVec are not implemented, and retail executables also require your console NAND to decrypt. See About → status. Use the Inspector to study this file.';
    return;
  }
  activateTab('run');
  const file = await library.getBlob(entry.id);
  const result = await runner.loadFile(new File([file], entry.name));
  const banner = $('run-banner');
  if (!result.ok) {
    banner.className = 'banner banner-error visible';
    banner.textContent = result.reason;
  } else {
    banner.className = 'banner banner-ok visible';
    banner.textContent = `Loaded ${entry.name}. Press Run to execute.`;
  }
}

/* ---------------- Dropzone & picker ---------------- */
const dropzone = $('dropzone');
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  if (e.dataTransfer?.files?.length) importFiles(e.dataTransfer.files);
});
$('add-files-btn').addEventListener('click', () => $('file-picker').click());
$('file-picker').addEventListener('change', (e) => {
  if (e.target.files?.length) importFiles(e.target.files);
  e.target.value = '';
});

/* ---------------- Runner & core ---------------- */
const runner = new Runner();
runner.setTrapOnUnimplemented(settings.trapOnUnimplemented);
runner.setBudget(settings.budget);

const runnerState = { prevGprs: new Map(), fps: 0, frames: 0, lastFpsAt: performance.now() };

function appendConsole(text) {
  const out = $('console-output');
  out.textContent += text;
  // Cap the console at ~64 KB to keep layout work bounded.
  if (out.textContent.length > 65536) {
    out.textContent = `…(trimmed)\n${out.textContent.slice(-49152)}`;
  }
  out.scrollTop = out.scrollHeight;
}

function renderRegisters(gpResolved) {
  const grid = $('register-grid');
  if (!grid.dataset.built) {
    for (let i = 0; i < 32; i++) {
      const cell = document.createElement('div');
      cell.className = 'reg-cell';
      cell.innerHTML = `<span class="reg-name">r${i}</span><span class="reg-value">0</span>`;
      cell.id = `reg-${i}`;
      grid.appendChild(cell);
    }
    grid.dataset.built = '1';
  }
  for (let i = 0; i < 32; i++) {
    const value = runner.core.gpr(i) & 0xffffffffffffffffn;
    const cell = $(`reg-${i}`);
    const prev = runnerState.prevGprs.get(i);
    cell.querySelector('.reg-value').textContent = value.toString(16).toUpperCase().padStart(16, '0').slice(-8);
    cell.classList.toggle('reg-hot', prev !== undefined && prev !== value);
    runnerState.prevGprs.set(i, value);
  }
}

function renderRunnerState(state) {
  $('state-pc').textContent = formatHex(state.pc);
  $('state-lr').textContent = formatHex(state.lr);
  $('state-ctr').textContent = formatHex(state.ctr);
  $('state-xer').textContent = formatHex(state.xer);
  $('state-cr').textContent = `0x${(state.cr >>> 0).toString(16).padStart(8, '0').toUpperCase()}`;
  $('state-insns').textContent = state.instructions.toLocaleString();
  $('state-unimpl').textContent = state.unimplemented.toLocaleString();
  $('state-mips').textContent = state.mips.toFixed(1);
  $('mem-loaded').textContent = `${formatBytes(state.loadedBytes)} at ${formatHex(BigInt(state.loadedBase), 8)}`;
  renderRegisters();
}

runner.onChange((kind, payload) => {
  if (kind === 'state') {
    renderRunnerState(payload);
  } else if (kind === 'console') {
    appendConsole(payload);
  } else if (kind === 'loaded') {
    appendConsole(`; loaded ${payload.name}\n`);
  } else if (kind === 'halted') {
    $('btn-run').disabled = false;
    $('btn-pause').disabled = true;
    $('btn-step').disabled = false;
    const banner = $('run-banner');
    if (payload.status === 1) {
      banner.className = 'banner banner-ok visible';
      banner.textContent = `Halted cleanly at ${formatHex(payload.pc)} after ${payload.instructions.toLocaleString()} instructions.`;
    } else if (payload.status === 2) {
      banner.className = 'banner banner-warn visible';
      banner.textContent = `Trapped on unimplemented instruction ${formatHex(BigInt(payload.lastUnimplemented), 8)} at ${formatHex(payload.pc)}. Disable the trap in Settings to skip VMX opcodes (they are not executed).`;
    } else if (payload.status === 3) {
      banner.className = 'banner banner-error visible';
      banner.textContent = `Memory fault at ${formatHex(payload.lastFaultAddress)} (PC ${formatHex(payload.pc)}). The guest dereferenced an address outside its 64 MiB RAM window.`;
    } else {
      banner.className = 'banner banner-info visible';
      banner.textContent = `status ${payload.status} at ${formatHex(payload.pc)}`;
    }
    diagLog(`halted: status=${payload.status} pc=${formatHex(payload.pc)} insns=${payload.instructions}`);
  } else if (kind === 'running') {
    $('btn-run').disabled = payload.running;
    $('btn-pause').disabled = !payload.running;
  } else if (kind === 'need-reload') {
    // Keep the message actionable in the console.
    appendConsole('; reset cleared guest RAM — press Run to reload and re-execute\n');
  }
});

function loadBuiltinIntoRunner(program) {
  const banner = $('run-banner');
  try {
    runner.loadSource(program.source, program.entry, program.name);
    banner.className = 'banner banner-ok visible';
    banner.textContent = `${program.name} assembled and loaded. Press Run.`;
    setStatusMsg(`loaded ${program.name}`);
    diagLog(`loaded builtin ${program.id}`);
  } catch (error) {
    banner.className = 'banner banner-error visible';
    banner.textContent = `Assembler error: ${error.message}`;
    toast(`Failed to assemble: ${error.message}`, 'error');
  }
}

function renderProgramList() {
  const list = $('program-list');
  list.innerHTML = '';
  for (const program of BUILTIN_PROGRAMS) {
    const item = document.createElement('button');
    item.className = 'program-item';
    item.innerHTML = `<div class="program-name">${escapeHtml(program.name)}</div><div class="program-desc">${escapeHtml(program.description)}</div>`;
    item.addEventListener('click', () => {
      $$('.program-item', list).forEach((el) => el.classList.remove('selected'));
      item.classList.add('selected');
      loadBuiltinIntoRunner(program);
    });
    list.appendChild(item);
  }
}

$('btn-run').addEventListener('click', () => {
  if (!runner.core || runner.isHalted() || true) runner.run();
});
$('btn-pause').addEventListener('click', () => runner.pause());
$('btn-step').addEventListener('click', () => { $('btn-run').disabled = false; $('btn-pause').disabled = true; runner.step(); });
$('btn-reset').addEventListener('click', () => {
  runner.reset();
  const banner = $('run-banner');
  banner.className = 'banner banner-info visible';
  banner.textContent = 'Guest state cleared (RAM zeroed). Press Run to reload the program.';
});
$('console-clear').addEventListener('click', () => { $('console-output').textContent = ''; });
$('run-load-file').addEventListener('click', () => $('run-file-picker').click());
$('run-file-picker').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  const banner = $('run-banner');
  try {
    const result = await runner.loadFile(file);
    if (result.ok) {
      banner.className = 'banner banner-ok visible';
      banner.textContent = `Loaded ${file.name}${result.kind === 'raw' ? ' as raw PPC at 0x10000' : ''}. Press Run to execute.`;
    } else {
      banner.className = 'banner banner-error visible';
      banner.textContent = result.reason;
    }
  } catch (error) {
    banner.className = 'banner banner-error visible';
    banner.textContent = `Failed to load: ${error.message}`;
  }
  e.target.value = '';
});

// Gamepad polling (informational until the input port lands).
setInterval(() => {
  const pads = navigator.getGamepads ? [...navigator.getGamepads()].filter(Boolean) : [];
  const el = $('gamepad-status');
  if (pads.length === 0) {
    el.textContent = 'No controllers detected. Press any button on a connected pad.';
  } else {
    el.textContent = `${pads.length} controller(s): ${pads.map((p) => p.id.slice(0, 40)).join(' · ')} — input port not implemented yet`;
  }
}, 750);

/* ---------------- FPS / perf overlay ---------------- */
function fpsLoop() {
  runnerState.frames++;
  const now = performance.now();
  if (now - runnerState.lastFpsAt >= 500) {
    runnerState.fps = (runnerState.frames * 1000) / (now - runnerState.lastFpsAt);
    runnerState.frames = 0;
    runnerState.lastFpsAt = now;
    const badge = $('fps-badge');
    const perf = $('status-perf');
    if (settings.perfOverlay) {
      badge.hidden = false;
      badge.textContent = `${runnerState.fps.toFixed(0)} FPS`;
      perf.hidden = false;
      perf.querySelector('.perf-fps').textContent = `${runnerState.fps.toFixed(0)} FPS`;
    } else {
      badge.hidden = true;
      perf.hidden = true;
    }
  }
  requestAnimationFrame(fpsLoop);
}
requestAnimationFrame(fpsLoop);

/* ---------------- Initialize core ---------------- */
(async function initCore() {
  const pill = $('core-pill');
  try {
    await runner.init();
    pill.className = 'pill pill-ok';
    pill.innerHTML = '<span class="dot"></span>Core: ready';
    $('status-core').textContent = 'wasm ok';
    diagLog('wasm core loaded');
    // Autoload the hello program so the Run tab is immediately usable.
    loadBuiltinIntoRunner(BUILTIN_PROGRAMS[0]);
    renderProgramList();
  } catch (error) {
    pill.className = 'pill pill-warn';
    pill.innerHTML = '<span class="dot"></span>Core: failed';
    $('status-core').textContent = 'core load error';
    toast(`Failed to load the WASM core: ${error.message}`, 'error');
    diagLog(`core load failed: ${error.message}`);
  }
})();

/* ---------------- Inspector ---------------- */
let inspectorFile = null;      // { name, blob } — from library or disk

function refreshInspectorSourceOptions() {
  const select = $('inspector-source');
  const current = select.value;
  select.innerHTML = '<option value="">Choose a library item…</option>';
  for (const entry of libraryEntries) {
    const option = document.createElement('option');
    option.value = entry.id;
    option.textContent = `${entry.name} (${KIND_LABEL[entry.kind] ?? 'BIN'}, ${formatBytes(entry.size)})`;
    select.appendChild(option);
  }
  select.value = current;
}

$('inspector-source').addEventListener('change', async (e) => {
  const id = e.target.value;
  if (!id) { inspectorFile = null; $('inspector-meta').innerHTML = '<p style="color:var(--text-dim);font-size:12px">Select a file to analyze.</p>'; return; }
  const entry = libraryEntries.find((x) => x.id === id);
  const blob = await library.getBlob(id);
  inspectorFile = { name: entry.name, blob, kind: entry.kind };
  await analyzeInspectorFile();
});

$('inspector-open-disk').addEventListener('click', () => $('inspector-file-picker').click());
$('inspector-file-picker').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  inspectorFile = { name: file.name, blob: file, kind: classifyMagic(new Uint8Array(await file.slice(0, 4).arrayBuffer())), offset: 0 };
  // Reset select to "custom" state.
  $('inspector-source').value = '';
  await analyzeInspectorFile();
  e.target.value = '';
});

async function analyzeInspectorFile() {
  if (!inspectorFile) return;
  const { blob, name } = inspectorFile;
  const BOUND = Math.min(blob.size, 8 * 1024 * 1024);
  const bytes = new Uint8Array(await blob.slice(0, BOUND).arrayBuffer());
  const kind = classifyMagic(bytes);
  $('disasm-offset').value = '0x10000';
  $('disasm-base').value = '0x10000';
  try {
    if (kind === 'xex') {
      const info = parseXex2(bytes);
      renderXexMeta(name, info, blob.size);
      $('disasm-offset').value = '0x0';
      $('disasm-base').value = info.imageBaseAddress ? formatHex(BigInt(info.imageBaseAddress), 8) : '0x82000000';
      $('disasm-run').click();
    } else if (kind === 'elf') {
      const info = parseElf(bytes);
      renderElfMeta(name, info, blob.size);
      const firstLoad = info.programHeaders.find((ph) => Number(ph.type) === 1);
      if (firstLoad) {
        $('disasm-offset').value = `0x${Number(firstLoad.offset).toString(16)}`;
        $('disasm-base').value = formatHex(firstLoad.vaddr, 8);
        $('disasm-run').click();
      }
    } else {
      renderBinMeta(name, bytes, blob.size);
      $('disasm-run').click();
    }
    diagLog(`inspected ${name} (${kind})`);
  } catch (error) {
    $('inspector-meta').innerHTML = `<table class="meta-table"><tr><th>Error</th><td>${escapeHtml(error.message)}</td></tr></table>`;
    diagLog(`inspect failed: ${error.message}`);
  }
}

function xexRow(th, td) { return `<tr><th>${escapeHtml(th)}</th><td>${escapeHtml(td)}</td></tr>`; }

function renderXexMeta(name, info, totalSize) {
  const pdTypes = info.pageDescriptors.map((p) => xexPageTypeName(p.type));
  const pdSummary = info.pageDescriptors.length
    ? `${info.pageDescriptors.length} pages (${countBy(pdTypes)}), ${formatBytes(info.pageDescriptors.reduce((a, p) => a + p.bytes, 0))}`
    : 'none';
  const flags = info.moduleFlagNames.length ? info.moduleFlagNames.join(', ') : '(none)';
  const rows = [
    ['File', `${name} (${formatBytes(totalSize)})`],
    ['Format', `${info.format} · header ${info.headerSize} B · ${info.headerCount} optional headers`],
    ['Module flags', flags],
    ['Entry point', info.entryPoint != null ? formatHex(BigInt(info.entryPoint), 8) : 'n/a'],
    ['Image base', info.imageBaseAddress != null ? formatHex(BigInt(info.imageBaseAddress), 8) : 'n/a'],
    ['Original base', info.originalBaseAddress != null ? formatHex(BigInt(info.originalBaseAddress), 8) : 'n/a'],
    ['Default stack', info.defaultStackSize != null ? formatBytes(info.defaultStackSize) : 'n/a'],
    ['Default heap', info.defaultHeapSize != null ? formatBytes(info.defaultHeapSize) : 'n/a'],
    ['Image size', info.imageSize != null ? formatBytes(info.imageSize) : 'n/a'],
    ['Load address', info.loadAddress != null ? formatHex(BigInt(info.loadAddress), 8) : 'n/a'],
    ['Security offset', info.securityOffset ? `0x${info.securityOffset.toString(16)}` : 'none'],
    ['Page descriptors', pdSummary],
    ['Compression', info.compression?.compressionName ?? 'n/a'],
    ['Import libraries', info.importLibraries.length ? info.importLibraries.join(', ') : 'none found'],
  ];
  const table = rows.map(([k, v]) => xexRow(k, v)).join('');
  const headers = info.headers.slice(0, 48).map((h) =>
    xexRow(`0x${h.key.toString(16).padStart(8, '0')} ${h.keyName}`, h.value != null ? `0x${(h.value >>> 0).toString(16).padStart(8, '0')}${h.offset != null ? ` (@0x${h.offset.toString(16)})` : ''}` : `offset 0x${h.offset?.toString(16)}`)
  ).join('');
  $('inspector-meta').innerHTML = `
    <table class="meta-table">${table}</table>
    <div class="panel-title" style="margin-top:14px">Optional headers (${info.headers.length})</div>
    <table class="meta-table">${headers || '<tr><th>—</th><td>none</td></tr>'}</table>`;
}

function countBy(items) {
  const map = new Map();
  for (const item of items) map.set(item, (map.get(item) ?? 0) + 1);
  return [...map.entries()].map(([k, v]) => `${v} ${k}`).join(', ');
}

function renderElfMeta(name, info, totalSize) {
  const phRows = info.programHeaders.slice(0, 12).map((ph) =>
    xexRow(`PH ${ELF_PT[Number(ph.type)] ?? `type ${Number(ph.type)}`}`, `vaddr ${formatHex(ph.vaddr, 8)} · filesz ${formatBytes(Number(ph.filesz))} · memsz ${formatBytes(Number(ph.memsz))}`)
  ).join('');
  const shRows = info.sections.filter((s) => s.name).slice(0, 24).map((s) =>
    xexRow(s.name, `type ${s.shType} · addr ${formatHex(s.addr, 8)} · size ${formatBytes(Number(s.size))}`)
  ).join('');
  $('inspector-meta').innerHTML = `
    <table class="meta-table">
      ${xexRow('File', `${name} (${formatBytes(totalSize)})`)}
      ${xexRow('Format', `${info.format} · ${info.endianness}-endian · ${info.machineName}`)}
      ${xexRow('Type', `${info.eType}`)}
      ${xexRow('Entry', formatHex(info.entry, 8))}
      ${xexRow('Program headers', String(info.programHeaders.length))}
      ${xexRow('Sections', String(info.sections.length))}
    </table>
    <div class="panel-title" style="margin-top:14px">Program headers</div>
    <table class="meta-table">${phRows || '<tr><th>—</th><td>none</td></tr>'}</table>
    <div class="panel-title" style="margin-top:14px">Sections</div>
    <table class="meta-table">${shRows || '<tr><th>—</th><td>no named sections</td></tr>'}</table>`;
}

function renderBinMeta(name, bytes, totalSize) {
  const magic = [...bytes.slice(0, 16)].map((b) => b.toString(16).padStart(2, '0')).join(' ');
  $('inspector-meta').innerHTML = `
    <table class="meta-table">
      ${xexRow('File', `${name} (${formatBytes(totalSize)})`)}
      ${xexRow('Format', 'Raw binary (no container)')}
      ${xexRow('First 16 bytes', magic)}
      ${xexRow('Hint', 'Use the disassembler with base 0x10000 for flat binaries, or the actual load address for embedded payloads.')}
    </table>`;
}

$('disasm-run').addEventListener('click', async () => {
  if (!inspectorFile) { toast('Open a file first.', 'warn'); return; }
  const out = $('disasm-output');
  const offset = parseInt($('disasm-offset').value.trim(), 0);
  const count = Math.min(2000, Math.max(1, Number($('disasm-count').value) || 64));
  const base = parseInt($('disasm-base').value.trim(), 0) >>> 0;
  if (!Number.isFinite(offset) || offset < 0) { toast('Bad offset', 'error'); return; }
  out.innerHTML = '';
  try {
    const slice = new Uint8Array(await inspectorFile.blob.slice(offset, offset + count * 4).arrayBuffer());
    const dv = new DataView(slice.buffer, slice.byteOffset, slice.byteLength);
    const rows = [];
    for (let i = 0; i + 4 <= slice.length; i += 4) {
      const code = dv.getUint32(i, false); // PPC is big-endian
      const vaddr = (base + i) >>> 0;
      const line = runner.core
        ? runner.core.disassembleLine(vaddr, code)
        : '(core not ready)';
      const group = runner.core ? runner.core.opcodeGroup(code) : 'unknown';
      const extra = group === 'unknown' ? ' unimplemented' : '';
      rows.push(`<div class="disasm-row${extra}"><span class="disasm-addr">${formatHex(BigInt(vaddr), 8)}</span><span class="disasm-hex">${code.toString(16).padStart(8, '0')}</span><span class="disasm-text">${escapeHtml(line)}</span></div>`);
    }
    out.innerHTML = rows.join('') || '<div class="disasm-row"><span class="disasm-text" style="color:var(--text-faint)">Nothing to disassemble at that offset.</span></div>';
  } catch (error) {
    out.innerHTML = `<div class="disasm-row"><span class="disasm-text" style="color:var(--red)">Disassembly failed: ${escapeHtml(error.message)}</span></div>`;
  }
});

function showBuiltinSource(program) {
  inspectorFile = null;
  $('inspector-source').value = '';
  const out = $('disasm-output');
  $('inspector-meta').innerHTML = `<table class="meta-table">
    ${xexRow('Program', program.name)}
    ${xexRow('Entry', formatHex(BigInt(program.entry), 8))}
    ${xexRow('Kind', 'built-in assembly (js/programs.js)')}
  </table>`;
  // Render the source as pseudo-disassembly lines.
  out.innerHTML = program.source.split('\n').map((line) =>
    `<div class="disasm-row"><span class="disasm-addr">asm</span><span class="disasm-hex">—</span><span class="disasm-text">${escapeHtml(line)}</span></div>`
  ).join('');
}

/* ---------------- Settings wiring ---------------- */
const settingTrap = $('setting-trap');
const settingBudget = $('setting-budget');
const settingHash = $('setting-hash');
const settingPerf = $('setting-perf');
const settingConfirmDelete = $('setting-confirm-delete');
const settingVerbose = $('setting-verbose');

settingTrap.checked = settings.trapOnUnimplemented;
settingBudget.value = String(settings.budget);
settingHash.checked = settings.fullHash;
settingPerf.checked = settings.perfOverlay;
settingConfirmDelete.checked = settings.confirmDelete;
settingVerbose.checked = settings.verbose;

settingTrap.addEventListener('change', () => { settings.trapOnUnimplemented = settingTrap.checked; runner.setTrapOnUnimplemented(settingTrap.checked); saveSettings(settings); });
settingBudget.addEventListener('change', () => { settings.budget = Number(settingBudget.value); runner.setBudget(settings.budget); saveSettings(settings); });
settingHash.addEventListener('change', () => { settings.fullHash = settingHash.checked; saveSettings(settings); toast('Applies to future imports.', 'ok'); });
settingPerf.addEventListener('change', () => { settings.perfOverlay = settingPerf.checked; saveSettings(settings); });
settingConfirmDelete.addEventListener('change', () => { settings.confirmDelete = settingConfirmDelete.checked; saveSettings(settings); });
settingVerbose.addEventListener('change', () => { settings.verbose = settingVerbose.checked; saveSettings(settings); });

$('clear-library').addEventListener('click', async () => {
  if (!library) return;
  const count = libraryEntries.length;
  if (!window.confirm(`Delete all ${count} dump(s) and their metadata? This cannot be undone.`)) return;
  await library.clear();
  toast('Library cleared.', 'ok');
  diagLog('library cleared by user');
  await refreshLibrary();
});

/* ---------------- About: status tables ---------------- */
const STATUS_IMPLEMENTED = [
  ['Xenon PPC decoder (full xenia-canary opcode table)', 'ok'],
  ['Disassembler (xenia opcode descriptors)', 'ok'],
  ['Interpreter: integer ALU, 64-bit ops, rotate/mask/shift', 'ok'],
  ['Loads/stores, branches/CTR/LK, sc syscall, MMIO console', 'ok'],
  ['Fixed 64 MiB guest RAM · bounded WASM linear memory (no OOM growth)', 'ok'],
  ['OPFS-backed dump library with streaming imports', 'ok'],
  ['XEX2 / ELF metadata inspection', 'ok'],
  ['Built-in Xenon assembly programs (assembler + runner)', 'ok'],
];
const STATUS_MISSING = [
  ['Xbox 360 kernel HLE (xboxkrnl: TLS, threading, heap, sync)', 'none'],
  ['Xenos GPU + WebGPU presenter', 'none'],
  ['AltiVec/VMX execution — traps as unimplemented', 'none'],
  ['LZX / XEX page decompression for retail titles', 'none'],
  ['Retail game boot (needs console NAND + CPU key — never shipped)', 'none'],
  ['Emulated audio (XMA), input stack, savegames', 'none'],
  ['Threaded JIT / SharedArrayBuffer core', 'none'],
  ['Floating-point correctness beyond basic arithmetic', 'partial'],
];
function renderStatusList(elId, rows) {
  const host = $(elId);
  host.innerHTML = rows.map(([text, chip]) => {
    const label = chip === 'ok' ? 'done' : chip === 'partial' ? 'partial' : 'missing';
    return `<div class="status-row"><span>${escapeHtml(text)}</span><span class="status-chip chip-${chip}">${label}</span></div>`;
  }).join('');
}
renderStatusList('status-ok-list', STATUS_IMPLEMENTED);
renderStatusList('status-missing-list', STATUS_MISSING);

/* ---------------- About: diagnostics ---------------- */
async function collectDiagnostics() {
  const wasmSize = await fetch('core/xeniatest-core.wasm', { method: 'HEAD' }).then((r) => r.headers.get('content-length')).catch(() => null);
  const snapshot = runner.coreSnapshot();
  return {
    ...diagnostics,
    wasmCoreBytes: wasmSize ? Number(wasmSize) : null,
    settings,
    storage: await library?.estimate().catch(() => null) ?? null,
    library: { count: libraryEntries.length, kinds: countBy(libraryEntries.map((e) => e.kind)) },
    core: snapshot,
    memoryBudgetMB: 88,
    timestamp: new Date().toISOString(),
  };
}

$('export-diagnostics').addEventListener('click', async () => {
  const data = await collectDiagnostics();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'xeniatest-diagnostics.json';
  a.click();
  URL.revokeObjectURL(a.href);
  diagLog('diagnostics exported');
});
$('copy-diagnostics').addEventListener('click', async () => {
  const data = await collectDiagnostics();
  try {
    await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    toast('Diagnostics copied to clipboard.', 'ok');
  } catch {
    toast('Clipboard unavailable — use Export instead.', 'warn');
  }
});

/* ---------------- Boot ---------------- */
refreshThreadsStatus();
refreshLibrary();
renderBuiltins();
refreshStorageStatus();
setStatusMsg('ready');
diagLog('app started');
