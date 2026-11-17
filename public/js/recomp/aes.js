/**
 * aes.js — minimal AES-128 block cipher, no dependencies.
 *
 * Implements the full AES-128 (FIPS-197) substitution/permutation network for
 * both encryption and decryption, plus the two modes the XEX2 loader needs:
 *
 *   - aesEcbDecrypt(key, block16)  — unwrap the XEX2 session key
 *   - aesCbcChain(sessionKey, input[, output, ivec]) — the XEX2 image format
 *     uses a single, image-wide AES-128-CBC chain with a zero IV (xenia's
 *     aes_decrypt_buffer). The BASIC compression path decrypts blocks that
 *     carry explicit sizes, so callers drive the chain block by block and
 *     pass the running `ivec` along; use aesCbcDecryptBlock for that.
 *
 * Verified against the FIPS-197 Appendix B / C.1 vectors and OpenSSL
 * (`openssl enc -aes-128-ecb/-cbc -d -nopad`) in core/tests.
 */

const SBOX = new Uint8Array([
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
]);

// Inverse S-box: INV_SBOX[SBOX[i]] === i
const INV_SBOX = new Uint8Array(256);
for (let i = 0; i < 256; i++) INV_SBOX[SBOX[i]] = i;

const RCON = new Uint8Array([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]);

/** Multiply in GF(2^8) with the AES polynomial x^8+x^4+x^3+x+1 (0x11b). */
function gmul(a, b) {
  let p = 0, x = a, y = b;
  while (y) {
    if (y & 1) p ^= x;
    x <<= 1;
    if (x & 0x100) x ^= 0x11b;
    y >>>= 1;
  }
  return p;
}

/** xtime: multiply by 2 in GF(2^8) with reduction by 0x1b. */
function xtime(a) { return ((a << 1) ^ (a & 0x80 ? 0x1b : 0)) & 0xff; }

/** Multiply by 3 in GF(2^8). */
function mul3(a) { return xtime(a) ^ a; }

/** AES-128 key expansion: returns 176 bytes (11 round keys). */
export function expandKey(key) {
  if (!(key instanceof Uint8Array) || key.length !== 16) throw new Error('AES-128 key must be 16 bytes');
  const w = new Uint8Array(176);
  w.set(key, 0);
  let bytesGenerated = 16;
  let rconIdx = 0;
  const temp = new Uint8Array(4);
  while (bytesGenerated < 176) {
    for (let i = 0; i < 4; i++) temp[i] = w[bytesGenerated - 4 + i];
    if (bytesGenerated % 16 === 0) {
      // RotWord
      const t = temp[0]; temp[0] = temp[1]; temp[1] = temp[2]; temp[2] = temp[3]; temp[3] = t;
      // SubWord
      for (let i = 0; i < 4; i++) temp[i] = SBOX[temp[i]];
      temp[0] ^= RCON[rconIdx++];
    }
    for (let i = 0; i < 4; i++) {
      w[bytesGenerated] = w[bytesGenerated - 16] ^ temp[i];
      bytesGenerated++;
    }
  }
  return w;
}

function addRoundKey(state, roundKey) {
  for (let i = 0; i < 16; i++) state[i] ^= roundKey[i];
}

function shiftRows(s) {
  // Row 1: shift left 1, row 2: left 2, row 3: left 3 (column-major state)
  let t;
  t = [s[5], s[9], s[13], s[1]];
  s[1] = t[0]; s[5] = t[1]; s[9] = t[2]; s[13] = t[3];
  t = [s[10], s[14], s[2], s[6]];
  s[2] = t[0]; s[6] = t[1]; s[10] = t[2]; s[14] = t[3];
  t = [s[15], s[3], s[7], s[11]];
  s[3] = t[0]; s[7] = t[1]; s[11] = t[2]; s[15] = t[3];
}

function invShiftRows(s) {
  let t;
  t = [s[13], s[1], s[5], s[9]];
  s[1] = t[0]; s[5] = t[1]; s[9] = t[2]; s[13] = t[3];
  t = [s[10], s[14], s[2], s[6]]; // row 2: shift right 2 == left 2
  s[2] = t[0]; s[6] = t[1]; s[10] = t[2]; s[14] = t[3];
  t = [s[7], s[11], s[15], s[3]];
  s[3] = t[0]; s[7] = t[1]; s[11] = t[2]; s[15] = t[3];
}

function subBytes(s) { for (let i = 0; i < 16; i++) s[i] = SBOX[s[i]]; }
function invSubBytes(s) { for (let i = 0; i < 16; i++) s[i] = INV_SBOX[s[i]]; }

function mixColumns(s) {
  for (let c = 0; c < 4; c++) {
    const i = c * 4;
    const a0 = s[i], a1 = s[i + 1], a2 = s[i + 2], a3 = s[i + 3];
    s[i] = xtime(a0) ^ mul3(a1) ^ a2 ^ a3;
    s[i + 1] = a0 ^ xtime(a1) ^ mul3(a2) ^ a3;
    s[i + 2] = a0 ^ a1 ^ xtime(a2) ^ mul3(a3);
    s[i + 3] = mul3(a0) ^ a1 ^ a2 ^ xtime(a3);
  }
}

function invMixColumns(s) {
  for (let c = 0; c < 4; c++) {
    const i = c * 4;
    const a0 = s[i], a1 = s[i + 1], a2 = s[i + 2], a3 = s[i + 3];
    s[i]     = gmul(0x0e, a0) ^ gmul(0x0b, a1) ^ gmul(0x0d, a2) ^ gmul(0x09, a3);
    s[i + 1] = gmul(0x09, a0) ^ gmul(0x0e, a1) ^ gmul(0x0b, a2) ^ gmul(0x0d, a3);
    s[i + 2] = gmul(0x0d, a0) ^ gmul(0x09, a1) ^ gmul(0x0e, a2) ^ gmul(0x0b, a3);
    s[i + 3] = gmul(0x0b, a0) ^ gmul(0x0d, a1) ^ gmul(0x09, a2) ^ gmul(0x0e, a3);
  }
}

/** Encrypt one 16-byte block in place (operates on a copy). */
export function aesEncryptBlock(key, block) {
  const rk = expandKey(key);
  const s = Uint8Array.from(block);
  addRoundKey(s, rk.subarray(0, 16));
  for (let round = 1; round < 10; round++) {
    subBytes(s); shiftRows(s); mixColumns(s);
    addRoundKey(s, rk.subarray(round * 16, round * 16 + 16));
  }
  subBytes(s); shiftRows(s);
  addRoundKey(s, rk.subarray(160, 176));
  return s;
}

/** Decrypt one 16-byte block in place (operates on a copy). */
export function aesDecryptBlock(key, block) {
  const rk = expandKey(key);
  const s = Uint8Array.from(block);
  addRoundKey(s, rk.subarray(160, 176));
  for (let round = 9; round >= 1; round--) {
    invShiftRows(s); invSubBytes(s);
    addRoundKey(s, rk.subarray(round * 16, round * 16 + 16));
    invMixColumns(s);
  }
  invShiftRows(s); invSubBytes(s);
  addRoundKey(s, rk.subarray(0, 16));
  return s;
}

/** ECB decrypt (exactly 16-byte inputs in the XEX2 path). */
export function aesEcbDecrypt(key, data) {
  const out = new Uint8Array(data.length);
  for (let off = 0; off < data.length; off += 16) {
    out.set(aesDecryptBlock(key, data.subarray(off, off + 16)), off);
  }
  return out;
}

/** ECB encrypt — used by tests to build synthetic fixtures. */
export function aesEcbEncrypt(key, data) {
  const out = new Uint8Array(data.length);
  for (let off = 0; off < data.length; off += 16) {
    out.set(aesEncryptBlock(key, data.subarray(off, off + 16)), off);
  }
  return out;
}

/**
 * One CBC decrypt step with a caller-managed IV/chain.
 * Returns the 16 plaintext bytes; `chain` (Uint8Array(16)) is updated to the
 * ciphertext block, so it can be threaded across non-contiguous regions
 * (the XEX2 BASIC format advances the source by data_size while the chain
 * continues). Pass a zeroed `chain` for a fresh image-wide CBC.
 */
export function aesCbcDecryptBlock(key, cipherBlock, chain) {
  const raw = aesDecryptBlock(key, cipherBlock);
  const out = new Uint8Array(16);
  for (let i = 0; i < 16; i++) {
    out[i] = raw[i] ^ chain[i];
    chain[i] = cipherBlock[i];
  }
  return out;
}

/**
 * AES-128-CBC decrypt with a zero IV over the whole buffer (xenia's
 * aes_decrypt_buffer). Input length is rounded down to whole blocks.
 */
export function aesCbcDecrypt(key, data) {
  const chain = new Uint8Array(16); // zero IV
  const out = new Uint8Array(data.length - (data.length % 16));
  for (let off = 0; off < out.length; off += 16) {
    out.set(aesCbcDecryptBlock(key, data.subarray(off, off + 16), chain), off);
  }
  return out;
}

/** AES-128-CBC encrypt with a zero IV (tests / fixture building only). */
export function aesCbcEncrypt(key, data) {
  const chain = new Uint8Array(16); // zero IV
  const out = new Uint8Array(data.length);
  for (let off = 0; off < data.length; off += 16) {
    const block = new Uint8Array(16);
    for (let i = 0; i < 16; i++) block[i] = data[off + i] ^ chain[i];
    const ct = aesEncryptBlock(key, block);
    out.set(ct, off);
    chain.set(ct);
  }
  return out;
}
