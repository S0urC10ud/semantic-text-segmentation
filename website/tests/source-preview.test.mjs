import { test } from 'node:test';
import assert from 'node:assert/strict';
import { regions, sampleText, totalBytes, buildSourceLines } from '../src/lib/composition-example.mjs';

test('composition widths and byte ranges match the source exactly', () => {
  assert.match(sampleText, /^[\x00-\x7f]*$/);
  assert.equal(totalBytes, Buffer.byteLength(sampleText));
  assert.equal(regions[0].start, 0);
  regions.forEach((region, i) => {
    assert.equal(sampleText.slice(region.start, region.end), region.text);
    if (i) assert.equal(region.start, regions[i - 1].end);
  });
  assert.equal(regions.at(-1).end, totalBytes);
});

test('unfolding preserves all source lines and marks a region within a line', () => {
  const lines = buildSourceLines(sampleText, regions);
  assert.equal(lines.map(line => line.parts.map(part => part.text).join('')).join('\n'), sampleText);
  const mixed = buildSourceLines('prefix VALUE suffix\n', [
    {start: 0, end: 7, color: 'blue'},
    {start: 7, end: 12, color: 'purple'},
    {start: 12, end: 20, color: 'blue'},
  ]);
  assert.deepEqual(mixed[0].parts.map(part => part.regionIndex), [0, 1, 2]);
  assert.equal(mixed[0].parts[1].text, 'VALUE');
});
