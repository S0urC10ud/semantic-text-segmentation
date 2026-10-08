(function (root) {
  'use strict';

  // Ranges refer to the normalized ASCII string used by the browser model.
  // Reject stale or incomplete payloads instead of drawing misleading offsets.
  function buildRegions(payload) {
    const text = String(payload?.text || '');
    if (!text.length) return [];
    if (/[^\x00-\x7f]/.test(text)) throw new Error('Composition requires normalized ASCII text.');
    const segments = payload?.segments;
    if (!Array.isArray(segments) || !segments.length) throw new Error('Missing segments.');
    let cursor = 0;
    const regions = segments.map(segment => {
      const { start, end } = segment;
      if (!Number.isInteger(start) || !Number.isInteger(end) || start !== cursor || end <= start || end > text.length) {
        throw new Error('Segments must cover the normalized text in order.');
      }
      cursor = end;
      return { ...segment, startByte: start, endByte: end, bytes: end - start, percent: (end - start) / text.length * 100 };
    });
    if (cursor !== text.length) throw new Error('Incomplete segment coverage.');
    return regions;
  }

  root.TypeSegComposition = { buildRegions };
  if (typeof module !== 'undefined' && module.exports) module.exports = { buildRegions };
})(globalThis);
