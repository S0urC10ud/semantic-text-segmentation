import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { assertNoPublicPdfs } from '../scripts/check-public-assets.mjs';

test('the embargo check rejects a PDF in a nested public directory', async () => {
  const directory = await mkdtemp(path.join(tmpdir(), 'typeseg-embargo-test-'));
  try {
    await writeFile(path.join(directory, 'index.html'), '<p>TypeSeg</p>');
    await assertNoPublicPdfs(directory);
    await mkdir(path.join(directory, 'papers'));
    await writeFile(path.join(directory, 'papers', 'draft.PDF'), 'test fixture');
    await assert.rejects(assertNoPublicPdfs(directory), /PDF publication is disabled/);
  } finally {
    // Delete only the disposable directory this test created.
    assert.equal(path.dirname(path.resolve(directory)), path.resolve(tmpdir()));
    assert.ok(path.basename(directory).startsWith('typeseg-embargo-test-'));
    await rm(directory, { recursive: true });
  }
});
