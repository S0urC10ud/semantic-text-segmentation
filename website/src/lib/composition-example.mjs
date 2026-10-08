// ASCII example with illustrative annotations; ranges match the visible source.
const chunks = [
  { label: 'Python', color: '#3976c5', detail: 'The surrounding Python script.', text:
`import base64
from pathlib import Path

# A text file can carry several languages.
sample_name = "inspection-demo"

powershell_source = """
` },
  { label: 'PowerShell', color: '#8655c9', detail: 'An embedded PowerShell region to inspect.', text:
`$files = Get-ChildItem -Path . -File
$files | Select-Object Name, Length
Write-Output "Inspection complete"
` },
  { label: 'Python', color: '#3976c5', detail: 'Python delimiters and a second string.', text:
`"""

# The encoded text below is a harmless example.
encoded_note = "` },
  { label: 'Base64', color: '#b84f87', detail: 'A visible encoding for a downstream decoder.', text:
'VGhpcyBpcyBhIGhhcm1sZXNzIGV4YW1wbGUgZm9yIGNvbnRlbnQtdHlwZSBzZWdtZW50YXRpb24u' },
  { label: 'Python', color: '#3976c5', detail: 'The surrounding script continues.', text:
`"

decoded_note = base64.b64decode(encoded_note).decode("utf-8")
print(sample_name, decoded_note)
` },
];

let offset = 0;
export const regions = chunks.map(chunk => {
  const start = offset;
  offset += new TextEncoder().encode(chunk.text).length;
  return { ...chunk, start, end: offset };
});
export const sampleText = chunks.map(chunk => chunk.text).join('');
export const totalBytes = offset;

/**
 * @param {string} text
 * @param {{ start: number, end: number, color: string }[]} annotations
 * @returns {{ number: number, parts: { regionIndex: number, text: string, color: string }[] }[]}
 */
export function buildSourceLines(text, annotations) {
  let start = 0;
  return text.split('\n').map((line, index) => {
    const end = start + line.length;
    const parts = annotations.flatMap((region, regionIndex) => {
      const from = Math.max(start, region.start);
      const to = Math.min(end, region.end);
      return to > from ? [{ regionIndex, text: text.slice(from, to), color: region.color }] : [];
    });
    start = end + 1;
    return { number: index + 1, parts };
  });
}
