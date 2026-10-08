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
  const encodedLine = lines.find(line => line.parts.some(part => part.regionIndex === 3));
  assert.deepEqual(encodedLine.parts.map(part => part.regionIndex), [2, 3, 4]);
  assert.equal(Buffer.from(regions[3].text, 'base64').toString(), 'This is a harmless example for content-type segmentation.');
});
