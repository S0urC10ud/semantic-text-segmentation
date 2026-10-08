import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { buildRegions } = require('../../viewers/content_type_segmentor_static/composition.js');

test('keeps repeated content types in source order and measures their byte ranges', () => {
  const result = buildRegions({ text: 'aaBBBBcc', segments: [
    { start: 0, end: 2, label_id: 8 },
    { start: 2, end: 6, label_id: 17 },
    { start: 6, end: 8, label_id: 8 },
  ] });
  assert.deepEqual(result.map(r => [r.label_id, r.startByte, r.endByte, r.bytes, r.percent]), [
    [8, 0, 2, 2, 25], [17, 2, 6, 4, 50], [8, 6, 8, 2, 25],
  ]);
});

test('empty input produces no regions', () => assert.deepEqual(buildRegions({ text: '', segments: [] }), []));
test('one region fills the entire bar, including newlines', () => {
  assert.equal(buildRegions({ text: 'a\nb', segments: [{ start: 0, end: 3, label_id: 8 }] })[0].percent, 100);
});
test('rejects overlapping, gapped, out-of-range, and incomplete predictions', () => {
  for (const segments of [
    [{ start: 1, end: 3 }], [{ start: 0, end: 4 }], [{ start: 0, end: 1 }],
    [{ start: 0, end: 2 }, { start: 1, end: 3 }], [{ start: 0, end: 1 }, { start: 2, end: 3 }],
  ]) assert.throws(() => buildRegions({ text: 'abc', segments }));
});
test('rejects non-ASCII text instead of claiming character offsets are byte offsets', () => {
  assert.throws(() => buildRegions({ text: 'é', segments: [{ start: 0, end: 1 }] }));
});
