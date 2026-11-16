/**
 * Web app smoke test — boots the real recompiler page (index.html +
 * recompiler-ui.js) in jsdom, recompiles the bundled Fibonacci sample fully
 * in-page, checks the console output and stats, and validates the project
 * .zip that the download button produces (structure + CRCs + manifest).
 * Also keeps the inspector coverage for synthetic XEX2/ELF images.
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
  '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm',
  '.json': 'application/json', '.md': 'text/markdown',
};

function localFileResponse(path) {
  const body = readFileSync(path);
  const ext = path.slice(path.lastIndexOf('.'));
  return new Response(body, {
    status: 200,
    headers: { 'content-type': MIME[ext] ?? 'application/octet-stream' },
  });
}

let dom;
let consoleErrors;
let capturedBlob = null;

before(async () => {
  const html = readFileSync(INDEX, 'utf8');
  dom = new JSDOM(html, {
    url: 'http://localhost:3000/',
    pretendToBeVisual: true,
    runScripts: 'outside-only',
  });

  consoleErrors = [];
  dom.window.addEventListener('error', (event) => consoleErrors.push(`error event: ${event.message}`));

  const fetchShim = async (url) => {
    const resolved = new URL(String(url), 'http://localhost:3000/');
    const rel = decodeURIComponent(resolved.pathname);
    const filePath = join(PUBLIC, rel === '/' ? 'index.html' : rel);
    if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
      return new Response('not found', { status: 404 });
    }
    return localFileResponse(filePath);
  };

  globalThis.fetch = fetchShim;
  const define = (name, value) => Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
  define('window', dom.window);
  define('document', dom.window.document);
  define('navigator', dom.window.navigator);
  define('HTMLElement', dom.window.HTMLElement);
  define('Blob', dom.window.Blob ?? Blob);
  define('File', dom.window.File ?? File);
  define('FileReader', dom.window.FileReader);
  define('TextDecoder', TextDecoder);
  define('TextEncoder', TextEncoder);
  define('DataView', DataView);
  define('performance', performance);
  // jsdom has no createObjectURL — capture the Blob instead.
  const urlApi = dom.window.URL;
  urlApi.createObjectURL = (blob) => { capturedBlob = blob; return 'blob:mock'; };
  urlApi.revokeObjectURL = () => {};
  globalThis.URL = urlApi;
  dom.window.HTMLAnchorElement.prototype.click = function click() { /* captured via createObjectURL */ };

  await import('../../public/js/recompiler-ui.js');
  await new Promise((resolve) => setTimeout(resolve, 50));
});

after(() => {
  delete globalThis.fetch;
  dom.window.close();
});

function waitFor(predicate, { timeout = 15000, step = 25 } = {}) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const value = predicate();
      if (value) return resolve(value);
      if (Date.now() - start > timeout) return reject(new Error('waitFor timed out'));
      setTimeout(tick, step);
    };
    tick();
  });
}

test('recompiling the Fibonacci sample in-page prints fib(25) = 75025', async () => {
  const chip = document.querySelector('.chip[data-sample="fibonacci"]');
  assert.ok(chip, 'fibonacci sample chip exists');
  chip.click();
  await waitFor(() => document.getElementById('image-info').textContent.includes('entry'));
  const btn = document.getElementById('btn-recompile');
  assert.equal(btn.disabled, false);
  btn.click();
  const consoleOut = await waitFor(() => {
    const el = document.getElementById('console-out');
    return el.textContent.includes('fib(25) = 75025') ? el : null;
  });
  assert.match(consoleOut.textContent, /fib\(25\) = 75025\n/);
  const stats = document.getElementById('stats-strip');
  assert.equal(stats.hidden, false);
  assert.match(stats.textContent, /fragments\s*\d+/);
  assert.match(stats.textContent, /exit\s*0/);
  assert.equal(document.getElementById('btn-download').disabled, false);
});

test('download button emits a valid project zip (CRCs, manifest, segments)', async () => {
  const { crc32 } = await import('../../public/js/recomp/zip.js');
  capturedBlob = null;
  document.getElementById('btn-download').click();
  assert.ok(capturedBlob, 'zip blob was produced');
  const buf = await new Promise((resolve, reject) => {
    const reader = new dom.window.FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result));
    reader.onerror = reject;
    reader.readAsArrayBuffer(capturedBlob);
  });
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  assert.equal(view.getUint32(0, true), 0x04034b50, 'zip magic');
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0; i--) {
    if (view.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  }
  assert.ok(eocd >= 0, 'EOCD present');
  const count = view.getUint16(eocd + 10, true);
  let off = view.getUint32(eocd + 16, true);
  const names = [];
  for (let i = 0; i < count; i++) {
    assert.equal(view.getUint32(off, true), 0x02014b50);
    const crc = view.getUint32(off + 16, true);
    const size = view.getUint32(off + 20, true);
    const nameLen = view.getUint16(off + 28, true);
    const extraLen = view.getUint16(off + 30, true);
    const commentLen = view.getUint16(off + 32, true);
    const localOff = view.getUint32(off + 42, true);
    const name = new TextDecoder().decode(buf.subarray(off + 46, off + 46 + nameLen));
    const lName = view.getUint16(localOff + 26, true);
    const lExtra = view.getUint16(localOff + 28, true);
    const data = buf.subarray(localOff + 30 + lName + lExtra, localOff + 30 + lName + lExtra + size);
    assert.equal(crc32(data), crc, 'crc ok: ' + name);
    names.push(name);
    off += 46 + nameLen + extraLen + commentLen;
  }
  for (const expected of ['README.md', 'manifest.json', 'index.html', 'runtime.js', 'guest-wasm.js',
    'guest.wasm', 'run.js', 'docs/KERNEL.md', 'kernel/.gitkeep', 'segment-0.bin', 'guest.bin']) {
    assert.ok(names.includes(expected), 'zip contains ' + expected);
  }
  const manifestEntry = names.indexOf('manifest.json');
  assert.ok(manifestEntry >= 0);
});

test('inspector parses a synthetic XEX2 and a big-endian ELF64 header', async () => {
  const imageSize = 0x1000;
  const headerSize = 4 + 4 + 4 + 4 + 4 + 8;
  const securityOffset = 0x100;
  const buf = new Uint8Array(0x400);
  const dv = new DataView(buf.buffer);
  const put = (off, s) => { for (let i = 0; i < s.length; i++) buf[off + i] = s.charCodeAt(i); };
  put(0, 'XEX2');
  dv.setUint32(4, 0x00000080, false);
  dv.setUint32(8, headerSize, false);
  dv.setUint32(12, securityOffset, false);
  dv.setUint32(16, 1, false);
  dv.setUint32(20, 0x00010100, false);
  dv.setUint32(24, 0x82000000, false);
  dv.setUint32(securityOffset, 0x104, false);
  dv.setUint32(securityOffset + 4, imageSize, false);
  const secInfo = securityOffset + 8 + 0x100;
  dv.setUint32(secInfo, 0x18, false);
  dv.setUint32(secInfo + 4, 0x00000400, false);
  dv.setUint32(secInfo + 8, 0x82000000, false);
  dv.setUint32(secInfo + 12, 1, false);
  dv.setUint32(secInfo + 16, (1 << 28) | 4, false);

  const { parseXex2, parseElf } = await import('../../public/js/inspect.js');
  const info = parseXex2(buf);
  assert.equal(info.format, 'XEX2');
  assert.equal(info.entryPoint, 0x82000000);
  assert.equal(info.imageSize, imageSize);
  assert.equal(info.pageDescriptors.length, 1);
  assert.ok(info.moduleFlagNames.includes('USER_MODE'));

  const elf = new Uint8Array(64);
  elf.set([0x7f, 0x45, 0x4c, 0x46, 2, 2, 1, 0]);
  const edv = new DataView(elf.buffer);
  edv.setUint16(16, 2, false);
  edv.setUint16(18, 21, false);
  edv.setBigUint64(24, 0x82000000n, false);
  edv.setUint16(54, 56, false);
  edv.setUint16(56, 0, false);
  edv.setUint16(58, 64, false);
  edv.setUint16(60, 0, false);
  const elfInfo = parseElf(elf);
  assert.equal(elfInfo.format, 'ELF64');
  assert.equal(elfInfo.entry, 0x82000000n);
});

test('recompiler page boots with no unexpected console errors', () => {
  assert.deepEqual(consoleErrors, []);
});
