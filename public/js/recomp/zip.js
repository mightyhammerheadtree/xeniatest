/**
 * zip.js — minimal, dependency-free ZIP archive writer (STORE method).
 *
 * The recompiler project zips hold small artifacts (a few tens of KiB of
 * wasm plus text files), so compression buys nothing and keeping the writer
 * synchronous and exact is a feature. CRC-32 is computed with the standard
 * polynomial; names are written UTF-8 with the 0x0800 flag.
 *
 * Format: local file header + data per entry, then the central directory,
 * then the end-of-central-directory record. Everything little-endian.
 */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const te = new TextEncoder();

function pushU16(view, offset, value) { view.setUint16(offset, value & 0xffff, true); }
function pushU32(view, offset, value) { view.setUint32(offset, value >>> 0, true); }

/** DOS date+time for `date` (defaults to now), as [time, date] uint16s. */
function dosStamp(date) {
  const d = date ?? new Date();
  const time = ((d.getHours() & 0x1f) << 11) | ((d.getMinutes() & 0x3f) << 5) | ((d.getSeconds() / 2) & 0x1f);
  let year = d.getFullYear();
  if (year < 1980) year = 1980;
  const dosDate = (((year - 1980) & 0x7f) << 9) | (((d.getMonth() + 1) & 0x0f) << 5) | (d.getDate() & 0x1f);
  return [time, dosDate];
}

/**
 * Build a ZIP archive in memory.
 * @param {Array<{name: string, data: Uint8Array}>} entries
 * @returns {Uint8Array} the complete .zip file bytes
 */
export function makeZip(entries, { now } = {}) {
  const [dosTime, dosDate] = dosStamp(now);
  const chunks = [];
  const central = [];
  let offset = 0;

  for (const entry of entries) {
    const nameBytes = te.encode(entry.name);
    const data = entry.data;
    const crc = crc32(data);

    const localHeader = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(localHeader.buffer);
    pushU32(lv, 0, 0x04034b50);
    pushU16(lv, 4, 20);          // version needed
    pushU16(lv, 6, 0x0800);      // flags: UTF-8 names
    pushU16(lv, 8, 0);           // method: STORE
    pushU16(lv, 10, dosTime);
    pushU16(lv, 12, dosDate);
    pushU32(lv, 14, crc);
    pushU32(lv, 18, data.length);
    pushU32(lv, 22, data.length);
    pushU16(lv, 26, nameBytes.length);
    pushU16(lv, 28, 0);          // extra field length
    localHeader.set(nameBytes, 30);

    chunks.push(localHeader, data);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(centralHeader.buffer);
    pushU32(cv, 0, 0x02014b50);
    pushU16(cv, 4, 20);          // version made by
    pushU16(cv, 6, 20);          // version needed
    pushU16(cv, 8, 0x0800);      // flags
    pushU16(cv, 10, 0);          // method: STORE
    pushU16(cv, 12, dosTime);
    pushU16(cv, 14, dosDate);
    pushU32(cv, 16, crc);
    pushU32(cv, 20, data.length);
    pushU32(cv, 24, data.length);
    pushU16(cv, 28, nameBytes.length);
    pushU16(cv, 30, 0);          // extra length
    pushU16(cv, 32, 0);          // comment length
    pushU16(cv, 34, 0);          // disk number
    pushU16(cv, 36, 0);          // internal attrs
    pushU32(cv, 38, 0);          // external attrs
    pushU32(cv, 42, offset);     // local header offset
    centralHeader.set(nameBytes, 46);
    central.push(centralHeader);

    offset += localHeader.length + data.length;
  }

  const centralStart = offset;
  let centralSize = 0;
  for (const c of central) { chunks.push(c); centralSize += c.length; }

  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  pushU32(ev, 0, 0x06054b50);
  pushU16(ev, 4, 0);             // disk number
  pushU16(ev, 6, 0);             // central dir disk
  pushU16(ev, 8, central.length);
  pushU16(ev, 10, central.length);
  pushU32(ev, 12, centralSize);
  pushU32(ev, 16, centralStart);
  pushU16(ev, 20, 0);            // comment length
  chunks.push(eocd);

  const total = chunks.reduce((sum, c) => sum + c.length, 0);
  const out = new Uint8Array(total);
  let p = 0;
  for (const c of chunks) { out.set(c, p); p += c.length; }
  return out;
}
