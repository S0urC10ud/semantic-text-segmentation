import { cp, mkdir, copyFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const source = path.resolve(root, '../viewers/content_type_segmentor_static');
const target = path.join(root, 'public/demo');
await mkdir(target, { recursive: true });
// Publish the existing, independent browser runtime under /demo/.
// Keep model blobs, omit unused training/export artifacts (over 30 MB).
await cp(source, target, {
  recursive: true,
  filter: (file) => !/(?:tmp_savedmodel|\.npz$|\.onnx$|\.py$|test_.*\.js$|CNAME$)/.test(file),
});
for (const name of (await readdir(source)).filter(name => /^(?:favicon|apple-touch-icon)/.test(name))) {
  await copyFile(path.join(source, name), path.join(root, 'public', name));
}
await copyFile(path.join(source, 'CNAME'), path.join(root, 'public/CNAME'));
