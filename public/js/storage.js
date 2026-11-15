/**
 * storage.js — browser-local game library backed by the Origin Private
 * File System (OPFS). Dumps never leave the device, never go through a
 * server, and are streamed to disk in 4 MiB chunks so multi-gigabyte ISOs
 * do not exhaust memory.
 *
 * Layout inside OPFS:
 *   xeniatest/library.json      — metadata index (array of entries)
 *   xeniatest/games/<id>        — raw dump bytes, exactly as imported
 */

export const MAX_FULL_HASH_BYTES = 64 * 1024 * 1024;   // full SHA-256 under this
export const QUICK_HASH_CHUNK = 2 * 1024 * 1024;     // first/last 2 MiB otherwise
const COPY_CHUNK = 4 * 1024 * 1024;                  // streaming write granularity

const LIBRARY_DIR = 'xeniatest';
const GAMES_DIR = 'games';
const INDEX_FILE = 'library.json';

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

export function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit++; }
  const digits = value >= 100 || unit === 0 ? 0 : value >= 10 ? 1 : 1;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

/** Detect the dump kind from magic bytes (falling back to extension). */
export async function detectKind(file) {
  const head = new Uint8Array(await file.slice(0, 0x802).arrayBuffer());
  if (head.length >= 4 && head[0] === 0x58 && head[1] === 0x45 && head[2] === 0x58 && head[3] === 0x32) return 'xex'; // 'XEX2'
  if (head.length >= 4 && head[0] === 0x7f && head[1] === 0x45 && head[2] === 0x4c && head[3] === 0x46) return 'elf'; // ELF
  if (head.length >= 0x10) {
    const isoMagic = 'MICROSOFT*XBOX*MEDIA';
    const off = 0x801;
    let match = head.length >= off + isoMagic.length;
    if (match) {
      for (let i = 0; i < isoMagic.length; i++) {
        if (head[off + i] !== isoMagic.charCodeAt(i)) { match = false; break; }
      }
    }
    if (match) return 'iso';
  }
  const name = (file.name || '').toLowerCase();
  if (name.endsWith('.xex')) return 'xex';
  if (name.endsWith('.elf')) return 'elf';
  if (name.endsWith('.iso') || name.endsWith('.000') || name.endsWith('.cso') || name.endsWith('.img')) return 'iso';
  if (name.endsWith('.bin')) return 'bin';
  return 'bin';
}

export class LibraryUnavailable extends Error {}

export class GameLibrary {
  constructor(root, gamesDir) {
    this.root = root;        // xeniatest/ directory handle
    this.gamesDir = gamesDir;
    this.entries = [];       // in-memory index, persisted to library.json
  }

  static async open() {
    if (!navigator.storage || !navigator.storage.getDirectory) {
      throw new LibraryUnavailable('Origin Private File System is not available in this browser. Use a current Chromium, Firefox or Safari.');
    }
    const opfs = await navigator.storage.getDirectory();
    const root = await opfs.getDirectoryHandle(LIBRARY_DIR, { create: true });
    const gamesDir = await root.getDirectoryHandle(GAMES_DIR, { create: true });
    const lib = new GameLibrary(root, gamesDir);
    lib.entries = await lib.readIndex();
    return lib;
  }

  async readIndex() {
    try {
      const handle = await this.root.getFileHandle(INDEX_FILE);
      const file = await handle.getFile();
      const parsed = JSON.parse(textDecoder.decode(await file.arrayBuffer()));
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  async writeIndex() {
    const handle = await this.root.getFileHandle(INDEX_FILE, { create: true });
    const writable = await handle.createWritable();
    await writable.write(textEncoder.encode(JSON.stringify(this.entries, null, 2)));
    await writable.close();
  }

  /** List all tracked games, newest first. */
  async list() {
    return [...this.entries].sort((a, b) => (b.importedAt ?? 0) - (a.importedAt ?? 0));
  }

  /** Persist a new file and stream it into OPFS. Returns the entry. */
  async importFile(file, { onProgress } = {}) {
    const id = (crypto.randomUUID ? crypto.randomUUID() : `game-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    const kind = await detectKind(file);
    const entry = {
      id,
      name: file.name || 'unnamed',
      size: file.size,
      kind,
      importedAt: Date.now(),
      sha256: null,
      quickHash: null,
      format: kind,
    };

    // Stream to disk in chunks.
    const dest = await this.gamesDir.getFileHandle(id, { create: true });
    const writable = await dest.createWritable();
    let written = 0;
    try {
      const reader = file.stream().getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        // value is already a bounded chunk; re-chunk to COPY_CHUNK for RAM discipline.
        await writable.write(value);
        written += value.byteLength;
        if (onProgress) onProgress(written, file.size);
      }
    } finally {
      await writable.close();
    }

    // Fingerprint (bounded memory).
    try {
      if (file.size <= MAX_FULL_HASH_BYTES && crypto.subtle) {
        const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
        entry.sha256 = toHex(new Uint8Array(digest));
      } else if (crypto.subtle) {
        entry.quickHash = await this.quickFingerprint(file);
      }
    } catch {
      // Fingerprinting is best-effort; the file is still imported.
    }

    this.entries.push(entry);
    await this.writeIndex();
    return entry;
  }

  async quickFingerprint(file) {
    const headEnd = Math.min(file.size, QUICK_HASH_CHUNK);
    const tailStart = Math.max(0, file.size - QUICK_HASH_CHUNK);
    const head = await file.slice(0, headEnd).arrayBuffer();
    const tail = tailStart === 0 && headEnd === file.size ? head : await file.slice(tailStart, file.size).arrayBuffer();
    const sizeBuf = new Uint8Array(8);
    new DataView(sizeBuf.buffer).setBigUint64(0, BigInt(file.size), false);
    const joined = new Uint8Array(head.byteLength + tail.byteLength + 8);
    joined.set(new Uint8Array(head), 0);
    joined.set(new Uint8Array(tail), head.byteLength);
    joined.set(sizeBuf, head.byteLength + tail.byteLength);
    const digest = await crypto.subtle.digest('SHA-256', joined);
    return toHex(new Uint8Array(digest));
  }

  /** Get a Blob for a stored dump (streamed from disk on read). */
  async getBlob(id) {
    const handle = await this.gamesDir.getFileHandle(id);
    return handle.getFile();
  }

  /** Read a bounded byte range — the inspector uses this on huge ISOs. */
  async readRange(id, offset, length) {
    const blob = await this.getBlob(id);
    const slice = blob.slice(offset, Math.min(offset + length, blob.size));
    return new Uint8Array(await slice.arrayBuffer());
  }

  async delete(id) {
    await this.gamesDir.removeEntry(id);
    this.entries = this.entries.filter((e) => e.id !== id);
    await this.writeIndex();
  }

  async clear() {
    const doomed = this.entries.map((e) => e.id);
    for (const id of doomed) {
      try { await this.gamesDir.removeEntry(id); } catch { /* already gone */ }
    }
    this.entries = [];
    await this.writeIndex();
  }

  async estimate() {
    if (navigator.storage && navigator.storage.estimate) {
      const { usage, quota } = await navigator.storage.estimate();
      return { used: usage ?? 0, quota: quota ?? 0 };
    }
    return { used: 0, quota: 0 };
  }
}

function toHex(bytes) {
  return [...bytes].map((b) => b.toString(16).padStart(2, '0')).join('');
}
