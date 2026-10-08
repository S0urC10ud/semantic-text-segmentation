import { readdir } from 'node:fs/promises';
import path from 'node:path';

// The paper is embargoed. Publishing a PDF must be an explicit future change.
export async function assertNoPublicPdfs(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) await assertNoPublicPdfs(file);
    else if (/\.pdf$/i.test(entry.name)) {
      throw new Error(`PDF publication is disabled while the paper is embargoed: ${file}`);
    }
  }
}
