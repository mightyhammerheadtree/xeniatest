/**
 * hle_bundle sync test — the exported project zip ships byte-identical
 * copies of hle.js / hle_tables.js / docs/HLE.md via the generated
 * public/js/recomp/hle_bundle.js. This fails if someone edits the sources
 * without regenerating (npm run build:hle-bundle).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { HLE_JS_SOURCE, HLE_TABLES_SOURCE, HLE_DOC_SOURCE } from '../../public/js/recomp/hle_bundle.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
const read = (p) => readFileSync(join(root, p), 'utf8');

test('hle_bundle.js embeds the exact hle.js source', () => {
  assert.equal(HLE_JS_SOURCE, read('public/js/recomp/hle.js'));
});

test('hle_bundle.js embeds the exact hle_tables.js source', () => {
  assert.equal(HLE_TABLES_SOURCE, read('public/js/recomp/hle_tables.js'));
});

test('hle_bundle.js embeds docs/HLE.md verbatim', () => {
  assert.equal(HLE_DOC_SOURCE, read('docs/HLE.md'));
});
