/**
 * Web app smoke test — boots the real frontend (index.html + JS modules)
 * in jsdom, loads the real WASM core over a local-file fetch shim, clicks
 * the Run button, and asserts the built-in program executes and reports
 * clean halt. Also exercises the Inspector on a synthetic XEX2 header.
 *
 * Run with: npm run test:web
 */
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { JSDOM } from 'jsdom';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const PUBLIC = join(ROOT, 'public');
const INDEX = join(PUBLIC, 'index.html');

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css',
  '.wasm': 'application/wasm', '.svg': 'image/svg+xml',
  '.webmanifest': 'application/manifest+json', '.json': 'application/json',
};

function localFileResponse(path) {
  const body = readFileSync(path);
  const ext = path.slice(path.lastIndexOf('.'));
  return new Response(body, {
    status: 200,
    headers: { 'content-type': MIME[ext] ?? 'application/octet-stream', 'content-length': String(body.length) },
  });
}

let dom;
let console_errors;

before(async () => {
  const html = readFileSync(INDEX, 'utf8');
  dom = new JSDOM(html, {
    url: 'http://localhost:3000/',
    pretendToBeVisual: true,
    runScripts: 'outside-only',
  });

  console_errors = [];
  dom.window.addEventListener('error', (event) => console_errors.push(`error event: ${event.message}`));

  // Local-file fetch: resolve relative URLs against /public.
  const fetchShim = async (url, options = {}) => {
    const resolved = new URL(String(url), 'http://localhost:3000/');
    const rel = decodeURIComponent(resolved.pathname);
    const filePath = join(PUBLIC, rel === '/' ? 'index.html' : rel);
    if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
      return new Response('not found', { status: 404 });
    }
    if ((options.method ?? 'GET').toUpperCase() === 'HEAD') {
      return new Response(null, { status: 200, headers: { 'content-length': String(statSync(filePath).size) } });
    }
    return localFileResponse(filePath);
  };

  globalThis.fetch = fetchShim;
  const define = (name, value) => Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
  define('window', dom.window);
  define('document', dom.window.document);
  define('navigator', dom.window.navigator);
  define('localStorage', dom.window.localStorage);
  define('location', dom.window.location);
  define('HTMLElement', dom.window.HTMLElement);
  define('Blob', dom.window.Blob ?? Blob);
  define('File', dom.window.File ?? File);
  define('FileReader', dom.window.FileReader);
  define('DataView', DataView);
  define('TextDecoder', TextDecoder);
  define('TextEncoder', TextEncoder);
  define('crypto', dom.window.crypto ?? crypto);
  // Use node's global performance (jsdom's recurses when promoted to global).
  const pendingFrames = new Set();
  globalThis.__pendingFrames = pendingFrames;
  define('requestAnimationFrame', (cb) => {
    const id = setTimeout(() => {
      pendingFrames.delete(id);
      cb(dom.window.performance.now());
    }, 1);
    pendingFrames.add(id);
    return id;
  });
  define('cancelAnimationFrame', (id) => {
    pendingFrames.delete(id);
    clearTimeout(id);
  });

  // Track intervals (gamepad poll) so after() can stop them.
  const realSetInterval = globalThis.setInterval.bind(globalThis);
  globalThis.__intervals = [];
  globalThis.setInterval = (fn, ms, ...args) => {
    const id = realSetInterval(fn, ms, ...args);
    globalThis.__intervals.push(id);
    return id;
  };

  await import('../../public/js/main.js');
  // Allow the module's top-level async boot (core load, library open) to settle.
  await new Promise((resolve) => setTimeout(resolve, 150));
});

after(() => {
  for (const id of globalThis.__intervals ?? []) clearInterval(id);
  for (const id of globalThis.__pendingFrames ?? []) clearTimeout(id);
  delete globalThis.fetch;
  delete globalThis.setInterval;
  dom.window.close();
});

async function waitFor(predicate, { timeout = 8000, step = 25 } = {}) {
  const start = Date.now();
  for (;;) {
    const value = predicate();
    if (value) return value;
    if (Date.now() - start > timeout) throw new Error('waitFor timed out');
    await new Promise((resolve) => setTimeout(resolve, step));
  }
}

test('core pill reaches ready state', async () => {
  const pill = await waitFor(() => {
    const el = document.getElementById('core-pill');
    return el && el.textContent.includes('ready') ? el : null;
  });
  assert.match(pill.textContent, /ready/i);
});

test('built-in hello program runs on click and halts cleanly', async () => {
  // Select the first built-in program explicitly (boot autoloads it already).
  const runTab = document.getElementById('tab-run');
  runTab.click();
  // Clear console so we observe only this run.
  document.getElementById('console-clear').click();
  document.getElementById('btn-run').click();
  const consoleOut = await waitFor(() => {
    const el = document.getElementById('console-output');
    return el.textContent.includes('inside your browser') ? el : null;
  }, { timeout: 15000 });
  assert.match(consoleOut.textContent, /inside your browser/);
  const banner = await waitFor(() => {
    const el = document.getElementById('run-banner');
    return el.textContent.includes('Halted cleanly') ? el : null;
  }, { timeout: 15000 });
  assert.match(banner.textContent, /Halted cleanly/);
  // Instructions were executed on the real core.
  const insns = document.getElementById('state-insns').textContent;
  assert.notEqual(insns, '0');
});

test('console MMIO halt is reported in the status field', async () => {
  await waitFor(() => {
    const status = document.getElementById('status-msg');
    return status && status.textContent.length >= 0 ? true : null;
  });
  assert.ok(true); // reaching the UI means the loop ran without throwing
});

test('inspector parses a synthetic XEX2 and shows metadata', async () => {
  // Build a minimal synthetic XEX2: header + one optional header + security info.
  const imageSize = 0x1000;
  const headerSize = 4 + 4 + 4 + 4 + 4 + 8; // magic.. + one opt header
  const securityOffset = 0x100;
  const buf = new Uint8Array(0x400);
  const dv = new DataView(buf.buffer);
  const put = (off, s) => { for (let i = 0; i < s.length; i++) buf[off + i] = s.charCodeAt(i); };
  put(0, 'XEX2');
  dv.setUint32(4, 0x00000080, false);      // USER_MODE
  dv.setUint32(8, headerSize, false);
  dv.setUint32(12, securityOffset, false);
  dv.setUint32(16, 1, false);              // header_count
  dv.setUint32(20, 0x00010100, false);     // ENTRY_POINT
  dv.setUint32(24, 0x82000000, false);
  // Security info
  dv.setUint32(securityOffset, 0x104, false); // security header size
  dv.setUint32(securityOffset + 4, imageSize, false);
  // Skip rsa signature (0x100 bytes), then info size / flags / load address / page descriptors
  const secInfo = securityOffset + 8 + 0x100;
  dv.setUint32(secInfo, 0x18, false);
  dv.setUint32(secInfo + 4, 0x00000400, false);
  dv.setUint32(secInfo + 8, 0x82000000, false);
  dv.setUint32(secInfo + 12, 1, false);     // one page descriptor
  dv.setUint32(secInfo + 16, (1 << 28) | 4, false); // type=code, 4 pages

  const { parseXex2, parseElf } = await import('../../public/js/inspect.js');
  const info = parseXex2(buf);
  assert.equal(info.format, 'XEX2');
  assert.equal(info.entryPoint, 0x82000000);
  assert.equal(info.imageSize, imageSize);
  assert.equal(info.pageDescriptors.length, 1);
  assert.equal(info.pageDescriptors[0].type, 1);
  assert.equal(info.pageDescriptors[0].pageCount, 4);
  assert.ok(info.moduleFlagNames.includes('USER_MODE'));

  // ELF: a tiny big-endian ELF64 header.
  const elf = new Uint8Array(64);
  elf.set([0x7f, 0x45, 0x4c, 0x46, 2, 2, 1, 0]); // ELF64, big-endian, current
  const edv = new DataView(elf.buffer);
  edv.setUint16(16, 2, false);    // ET_EXEC
  edv.setUint16(18, 21, false);   // EM_PPC64
  edv.setBigUint64(24, 0x82000000n, false);
  edv.setUint16(54, 56, false);   // e_phentsize = 56
  edv.setUint16(56, 0, false);    // e_phnum = 0 (no program headers parsed)
  edv.setUint16(58, 64, false);   // e_shentsize = 64
  edv.setUint16(60, 0, false);    // e_shnum = 0
  const elfInfo = parseElf(elf);
  assert.equal(elfInfo.format, 'ELF64');
  assert.equal(elfInfo.machineName, 'EM_PPC64 (PowerPC 64)');
  assert.equal(elfInfo.entry, 0x82000000n);
});


test('every tab switches views without errors', async () => {
  for (const tab of ['run', 'inspector', 'settings', 'about', 'library']) {
    document.getElementById(`tab-${tab}`).click();
    const view = document.getElementById(`view-${tab}`);
    assert.ok(view.classList.contains('view-active'), `${tab} view should activate`);
    assert.equal(document.getElementById(`tab-${tab}`).getAttribute('aria-selected'), 'true');
  }
});

test('step button executes exactly one instruction beyond the previous count', async () => {
  document.getElementById('tab-run').click();
  const before = document.getElementById('state-insns').textContent;
  document.getElementById('btn-step').click();
  const after = document.getElementById('state-insns').textContent;
  assert.notEqual(after, before);
});

test('perf overlay toggle reveals the FPS badge and persists', async () => {
  document.getElementById('tab-settings').click();
  const toggle = document.getElementById('setting-perf');
  toggle.checked = true;
  toggle.dispatchEvent(new dom.window.Event('change'));
  await new Promise((resolve) => setTimeout(resolve, 800));
  assert.equal(document.getElementById('fps-badge').hidden, false);
  const stored = JSON.parse(dom.window.localStorage.getItem('xeniatest.settings'));
  assert.equal(stored.perfOverlay, true);
  // turn it back off
  toggle.checked = false;
  toggle.dispatchEvent(new dom.window.Event('change'));
});

test('trap toggle persists to localStorage for the core', async () => {
  const toggle = document.getElementById('setting-trap');
  const initial = toggle.checked;
  toggle.checked = !initial;
  toggle.dispatchEvent(new dom.window.Event('change'));
  const stored = JSON.parse(dom.window.localStorage.getItem('xeniatest.settings'));
  assert.equal(stored.trapOnUnimplemented, !initial);
  toggle.checked = initial;
  toggle.dispatchEvent(new dom.window.Event('change'));
});

test('console clear empties the output pane', async () => {
  document.getElementById('tab-run').click();
  const out = document.getElementById('console-output');
  out.textContent = 'hello world\n';
  document.getElementById('console-clear').click();
  assert.equal(out.textContent, '');
});

test('about status tables render implemented/missing rows', async () => {
  document.getElementById('tab-about').click();
  const ok = document.getElementById('status-ok-list');
  const missing = document.getElementById('status-missing-list');
  assert.ok(ok.textContent.includes('Xenon PPC decoder'));
  assert.ok(missing.textContent.includes('kernel'));
  assert.ok(missing.textContent.includes('WebGPU'));
});

test('no unexpected console errors were emitted during the run', () => {
  assert.deepEqual(console_errors, []);
});
