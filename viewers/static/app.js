/* Multiple-example Segment Viewer
   - Renders each example as its own card (textarea + output + stats)
   - Keeps tooltip + legend behavior
   - Adds "Segment All" in the header
*/

let LABELS = [];
let STATE = []; // per-example state: { lastJson: null }
let tooltip = null;
let focusMode = false;
const MODEL_WINDOW = 1536;

const SANITIZE_REGEX = /[^\x20-\x7E¤\n\r\t]/g;

function sanitizeToCurrencySymbol(text){
  if (typeof text !== 'string' || text.length === 0){
    return text;
  }
  return text.replace(SANITIZE_REGEX, '¤');
}

function enforceSanitizedTextarea(textarea){
  if (!textarea){
    return;
  }
  let start = null;
  let end = null;
  try{
    start = textarea.selectionStart;
    end = textarea.selectionEnd;
  }catch(_err){
    // Accessing selection can throw if element is not focusable yet; ignore.
  }
  const sanitized = sanitizeToCurrencySymbol(textarea.value);
  if (sanitized !== textarea.value){
    textarea.value = sanitized;
    if (typeof start === 'number' && typeof end === 'number' && textarea.setSelectionRange){
      try{
        textarea.setSelectionRange(start, end);
      }catch(_err){
        // Some browsers require focus; ignore.
      }
    }
  }
}

function el(sel){ return document.querySelector(sel) }
function els(sel, root=document){ return Array.from(root.querySelectorAll(sel)) }
function esc(s){ return s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;') }

function showTooltip(event) {
  const target = event.target;
  if (!target || !target.classList || !target.classList.contains('char')) {
    hideTooltip();
    return;
  }
  if (!tooltip) tooltip = el('#tooltip');

  const probs = JSON.parse(target.dataset.probs || '{}');
  let html = '';

  const sortedProbs = Object.entries(probs)
    .map(([key, prob]) => {
      const idx = Number.parseInt(key, 10);
      const fromLabels = Number.isFinite(idx) ? LABELS[idx] : null;
      const label = fromLabels?.name || String(key);
      const color = fromLabels?.color || '#888888';
      return { id: Number.isFinite(idx) ? idx : key, prob, label, color };
    })
    .sort((a, b) => b.prob - a.prob);

  sortedProbs.forEach(({label, prob, color}) => {
    const percentage = (prob * 100).toFixed(1);
    html += `
      <div class="prob-bar">
        <div class="label">${esc(label)}</div>
        <div class="bar">
          <div class="fill" style="width:${percentage}%; --color:${color}"></div>
        </div>
        <div class="value">${percentage}%</div>
      </div>
    `;
  });

  tooltip.innerHTML = html || '<div class="prob-bar"><div class="label">No data</div></div>';
  tooltip.style.display = 'block';

  const rect = target.getBoundingClientRect();
  const tooltipRect = tooltip.getBoundingClientRect();
  let left = rect.left;
  let top = rect.bottom + 8;

  if (left + tooltipRect.width > window.innerWidth) {
    left = window.innerWidth - tooltipRect.width - 8;
  }
  if (top + tooltipRect.height > window.innerHeight) {
    top = rect.top - tooltipRect.height - 8;
  }

  tooltip.style.left = left + 'px';
  tooltip.style.top = top + 'px';
}

function hideTooltip() {
  if (tooltip) tooltip.style.display = 'none';
}

function buildLegend(holder){
  holder.innerHTML = '';
  LABELS.forEach(l => {
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.innerHTML = `<span class="dot" style="background:${l.color}"></span>${esc(l.name)}`;
    holder.appendChild(chip);
  });
}

function renderStatsInto(holder, stats){
  holder.innerHTML = '';
  (stats||[]).forEach(s => {
    const div = document.createElement('div');
    div.className = 'stat';
    div.innerHTML = `<span class="dot" style="background:${s.color}"></span>
      <b>${esc(s.name)}</b>
      <span>${s.count} chars</span>
      <div class="bar"><i style="width:${s.pct.toFixed(1)}%; background:${s.color}"></i></div>
      <span>${s.pct.toFixed(1)}%</span>`;
    holder.appendChild(div);
  });
}

async function runOne(section){
  const idx = parseInt(section.dataset.index, 10);
  const codeEl = section.querySelector('.code');
  const runBtn = section.querySelector('.runBtn');
  const minRun = parseInt(section.querySelector('.minRun').value || '6', 10);
  const chunk = MODEL_WINDOW;
  const render = section.querySelector('.render');
  const stats = section.querySelector('.stats');
  const timeLabel = section.querySelector('.inferenceTime');

  enforceSanitizedTextarea(codeEl);

  runBtn.disabled = true;
  const orig = runBtn.textContent;
  runBtn.textContent = 'Running…';
  if (timeLabel) {
    timeLabel.textContent = '…';
  }

  try{
    const res = await fetch('/api/segment', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        text: codeEl.value,
        min_run: minRun,
        chunk
      })
    });
    const data = await res.json();
    if (!res.ok){
      const detail = data && typeof data === 'object' ? (data.detail || data.error) : null;
      throw new Error(detail || `Request failed (${res.status})`);
    }
    STATE[idx].lastJson = data;
    render.innerHTML = data.html || '';
    renderStatsInto(stats, data.stats || []);
    if (timeLabel) {
      const elapsed = typeof data.elapsed_ms === 'number' ? data.elapsed_ms : null;
      if (elapsed !== null && isFinite(elapsed)) {
        timeLabel.textContent = elapsed >= 1000 ? (elapsed / 1000).toFixed(2) + ' s' : elapsed.toFixed(1) + ' ms';
      } else {
        timeLabel.textContent = '—';
      }
    }
  }catch(err){
    alert('Error: ' + err.message);
    if (timeLabel) {
      timeLabel.textContent = 'Err';
    }
  }finally{
    runBtn.disabled = false;
    runBtn.textContent = orig;
  }
}

async function runAll(){
  const sections = els('section.panel.example').filter(sec => !sec.hasAttribute('aria-hidden'));
  for (const s of sections){
    await runOne(s);
  }
}

function copyHTML(section){
  const render = section.querySelector('.render');
  const btn = section.querySelector('.copyHtml');
  const html = render.innerHTML;
  navigator.clipboard.writeText(html).then(()=>{
    btn.textContent = 'Copied!';
    setTimeout(()=> btn.textContent = 'Copy HTML', 800);
  });
}

function downloadJSON(section){
  const idx = parseInt(section.dataset.index, 10);
  const data = STATE[idx].lastJson || {};
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `segmentation-${idx+1}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function applyFocusMode(scrollToPrimary = false){
  const toggle = el('#focusToggle');
  focusMode = !!(toggle && toggle.checked);
  document.body.classList.toggle('focus-mode', focusMode);

  const sections = els('section.panel.example');
  sections.forEach((section, idx) => {
    const isPrimary = idx === 0;
    if (isPrimary) {
      section.classList.add('primary-example');
    }
    if (focusMode) {
      if (!isPrimary) {
        section.setAttribute('aria-hidden', 'true');
      } else {
        section.removeAttribute('aria-hidden');
      }
    } else {
      section.removeAttribute('aria-hidden');
    }
  });

  const label = el('#focusToggleLabel .toggle-text');
  if (label) {
    label.textContent = focusMode ? 'Show All Examples' : 'Focus Example 1';
  }
  const labelContainer = el('#focusToggleLabel');
  if (labelContainer) {
    labelContainer.classList.toggle('active', focusMode);
  }

  const primary = sections[0];
  if (primary) {
    if (focusMode) {
      primary.setAttribute('tabindex', '-1');
      if (scrollToPrimary) {
        primary.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    } else {
      primary.removeAttribute('tabindex');
    }
  }
}

function createExampleSection(index, initialText){
  const section = document.createElement('section');
  section.className = 'panel example';
  section.dataset.index = String(index);
  if (index === 0){
    section.classList.add('primary-example');
  }
  section.innerHTML = `
    <div class="panel-head">
      <h2>Example ${index+1}</h2>
      <div class="controls">
        <label>Min run (chars)
          <input class="minRun" type="number" min="1" step="1" value="4" />
        </label>
        <label>Window (bytes)
          <input class="chunk" type="number" min="64" step="64" value="1536" readonly />
        </label>
        <span class="inferenceTime" aria-live="polite" title="Wall-clock inference time">—</span>
        <button class="btn primary runBtn">Segment</button>
        <button class="btn copyHtml">Copy HTML</button>
        <button class="btn downloadJson">Download JSON</button>
      </div>
    </div>
    <div class="body">
      <div class="editor-pane">
        <textarea class="code" spellcheck="false" placeholder="Paste code/text (HTML/CSS/JS/C/CPP/CSV/Java/JSON/Python/Text)…"></textarea>
      </div>
      <div class="output-pane">
        <div class="legend"></div>
        <div class="render" aria-live="polite"></div>
        <div class="stats"></div>
      </div>
    </div>
  `;

  const codeEl = section.querySelector('.code');
  codeEl.value = sanitizeToCurrencySymbol(initialText);
  codeEl.addEventListener('input', ()=> enforceSanitizedTextarea(codeEl));
  codeEl.addEventListener('blur', ()=> enforceSanitizedTextarea(codeEl));

  section.querySelector('.runBtn').addEventListener('click', ()=> runOne(section));
  section.querySelector('.copyHtml').addEventListener('click', ()=> copyHTML(section));
  section.querySelector('.downloadJson').addEventListener('click', ()=> downloadJSON(section));

  return section;
}

async function bootstrap(){
  document.addEventListener('mousemove', showTooltip);
  document.addEventListener('mouseleave', hideTooltip);

  // Load labels / device info
  try{
    const res = await fetch('/api/labels');
    const meta = await res.json();
    LABELS = meta.labels || [];
    el('#device').textContent = (meta.loaded ? 'Loaded ✓ ' : 'Load error ✕ ') + (meta.device || '');

    // Build cards
    const holder = el('#examples');
    holder.innerHTML = '';

    EXAMPLES.forEach((text, i) => {
      STATE[i] = { lastJson: null };
      const sec = createExampleSection(i, text);
      holder.appendChild(sec);
      buildLegend(sec.querySelector('.legend'));
    });

  }catch(err){
    el('#device').textContent = 'Load error ✕';
    console.error(err);
  }

  const focusToggle = el('#focusToggle');
  if (focusToggle){
    focusToggle.addEventListener('change', ()=> applyFocusMode(focusToggle.checked));
  }
  applyFocusMode(focusToggle ? focusToggle.checked : false);

  // Segment All
  const runAllBtn = el('#runAll');
  if (runAllBtn){
    runAllBtn.addEventListener('click', runAll);
  }
}

/* === The "sketchy" examples, each rendered in its own editable card === */
// 30 language-diverse, slightly tricky examples covering:
// html, css, javascript, c_family, csv, java, json, python, text
const EXAMPLES = [
  `<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: system-ui; margin: 2rem; }
    .btn { background: #3498db; color: white; padding: 8px 12px; border-radius: 8px; }
    /* comment */ h1 { color: #e67e22; }
  </style>
  <script>
    const greet = (name) => console.log('hi', name);
    document.addEventListener('DOMContentLoaded', () => greet('world'));
  </script>
</head>
<body>
  <h1>Hello</h1>
  <button class="btn" onclick="alert('clicked')">Click</button>
</body>
</html>`,

`<!doctype html>
<html>
<head>
  <title>Commenty Script</title>
</head>
<body>
  <!-- Old browsers once needed these comment guards -->
  <script>
  <!--
  const payload = "<img src=x onerror=console.log('XSS?')>";
  // The HTML comment tokens shouldn't affect JS parsing now:
  console.log("ok"); //-->
  // String contains </script> safely when broken:
  const html = "</scr" + "ipt>";
  //-->
  </script>
  <!-- Above, the comment markers live *inside* the script -->
</body>
</html>`,

`<!doctype html><meta charset="utf-8">
<button
  <!-- comment in the middle of attributes -->
  class="cta"
  oncli<!--sneaky-->ck="console.log('clicked')"
>
  Click
</button>`,

`<!doctype html>
<style>
/* Comment splitting to dodge naive scanners */
.bg{
  background-image:
    url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg'/>");
  /* url("javascript:alert(1)")  — browsers block, but good test data */
}
/* @im/*trick*/*/port url("https://example.com/a.css"); /* broken tokenizing */
</style>
<div class="bg">Test</div>`,

`<!doctype html>
<svg width="10" height="10"
     onlo<!--split-->ad="console.log('svg onload')"
     xmlns="http://www.w3.org/2000/svg">
  <!-- foreignObject allows HTML inside SVG -->
  <foreignObject width="10" height="10">
    <div xmlns="http://www.w3.org/1999/xhtml" style="color:red">Hi</div>
  </foreignObject>
</svg>`,

`<!doctype html>
<style>
  /* harmless CSS */
  .x{color:#e67e22}
</style>
<script>
  // Put "</script>" inside a JS string safely by breaking it:
  const s = "</scr" + "ipt>";
  /* multiline
     comment */ console.log('ok');
</script>`,

`<!doctype html>
<!-- Normal users see nothing special -->
<!--[if IE]>
  <script>
    // Only old IE used to execute this
    console.log("IE branch");
  </script>
<![endif]-->
<p>Hello</p>`,

`<!doctype html>
<style>
/* Start CSS ... */
body{background:#111}
/* switch to something JS-looking but still CSS comment-wrapped:
   ;(() => { console.log('not executed, still CSS comment'); })();
*/
</style>
<script>
/*
  Start JS; embed CSS-like tokens that must remain JS comments:
  .x{background:url(#)}
*/
console.log("real JS");
</script>`,

`<!doctype html>
<script type="application/json">
{
  /* comments are not valid JSON, but some pipelines strip them */
  "cfg": {"safe": true},
  "msg": "hello"
}
</script>
<script>
  // JS reads previous tag’s textContent:
  const raw = document.querySelector('script[type="application/json"]').textContent;
  console.log('cfg len', raw.length);
</script>`,

/* ---------- C (3) ---------- */
`#include <stdio.h>
int main(void){
  const char *tag = "</scr" "ipt>"; /* concatenated literals */
  unsigned char u = 255;
  if (u == (unsigned char)-1) puts(tag);
  /* String with backslash-newline (line splicing): */
  const char *s = "hello, "\
                  "world";
  printf("%s\n", s);
  return 0;
}`,

`#include <stdio.h>
#define STR(x) #x
#define CONCAT(a,b) a b
#define DO(times, stmt) do{ for(int i=0;i<(times);++i){ stmt; } }while(0)
static int add(int a, int b){ return a+b; }
int main(void){
  int (*op)(int,int) = &add;
  DO(2, printf("%s\n", STR(not a // comment))); /* # makes it a string */
  printf("%d\n", op(2,3));
  puts(CONCAT("he","llo")); /* adjacent string-literal concat */
  return 0;
}`,

`#include <stdint.h>
#include <stdio.h>
struct Flags { unsigned a:1, b:2, c:1; };
int main(void){
  union { uint32_t u; unsigned char b[4]; } x = { .u = 0x01020304 };
  struct Flags f = (struct Flags){ .a=1, .b=3, .c=0 };
  printf("%02x %u\n", x.b[0], f.b);
  return 0;
}`,

/* ---------- C++ (3) ---------- */
`#include <iostream>
int main(){
  const char* html = R"(<!doctype html><title>ok</title><script>console.log("hi")</script>)";
  std::cout << html << '\n';
  return 0;
}`,

`#include <iostream>
#include <type_traits>
template<typename T>
concept Addable = requires(T a){ a + a; };
template<Addable T>
T twice(T x){ return x + x; }
int main(){
  std::cout << twice(21) << '\n';
  std::cout << twice(std::string("na")) << '\n';
}`,

`#include <map>
#include <string>
#include <iostream>
int main(){
  std::map<int,std::string> m{{1,"one"},{2,"two"}};
  for (auto [k,v] : m) std::cout << k << "=" << v << '\n';
  auto print = [](auto&& t){ std::cout << t << '\n'; };
  print(3.14);
}`,

/* ---------- CSV (3) ---------- */
`\uFEFFid,name,notes
1,"Doe, Jane","Line1
""Quoted"" value"
2,=2+3,"starts with formula"`,

`id;name;comment
3;Alice;"uses ; semicolons as delimiter"
4;Bob;"UTF-8 snowman ☃"`,

`"sku","title","price","tags"
"001","Chair ""Deluxe""","19.99","home,furniture"
"002","New
Line","0",""`,

/* ---------- Java (3) ---------- */
`package demo;
public class Main {
  public static void main(String[] args){
    String s = """
      { "json": true, "note": "inside Java text block" }
      """;
    System.out.println(s.strip());
  }
}`,

`import java.util.*;
public class App {
  @SafeVarargs
  static <T> List<T> listOf(T... items){ return Arrays.asList(items); }
  public static void main(String[] args){
    var xs = listOf(1,2,3);
    xs.replaceAll(n -> n+1);
    System.out.println(xs);
  }
}`,

`public record Point(int x, int y) {
  public String toString(){ return x + "," + y; }
  public static void main(String[] args){
    var p = new Point(1,2);
    System.out.println(p);
  }
}`,

/* ---------- JSON (3) ---------- */
`{
  "path": "C:\\\\temp\\\\file.json",
  "emoji": "\\uD83D\\uDE03",
  "numbers": [0, -0, 1e-3, 1E+2],
  "empty": null,
  "bool": true
}`,

`{
  "weird key!": "value",
  "list": ["a", "b", "c"],
  "nested": { "a": { "b": { "c": 1 } } }
}`,

`{
  "id": "9007199254740993",
  "url": "https:\\/\\/example.com\\/a\\/b",
  "text": "line1\\nline2",
  "flags": [true, false, null]
}`,

/* ---------- Python (3) ---------- */
`# -*- coding: utf-8 -*-
from __future__ import annotations

def greet(name: str) -> str:
    return f"hi, {name!r}".upper()

if __name__ == "__main__":
    who = "world"
    print(greet(who.replace("{", "{{")))`,

`from pathlib import Path

p = Path("data.txt")
with p.open("w", encoding="utf-8") as f:
    print("hello", file=f)

# assignment expression + match (3.10+)
n = len(p.read_bytes())
match n:
    case 0: print("empty")
    case _ if n % 2 == 0: print("even", n)
    case _: print("odd", n)`,

`def memo(fn):
    cache = {}
    def wrapper(x):
        if x in cache: return cache[x]
        cache[x] = fn(x)
        return cache[x]
    return wrapper

@memo
def fib(n: int) -> int:
    return n if n < 2 else fib(n-1) + fib(n-2)

nums = (fib(i) for i in range(10))
print(",".join(str(x) for x in nums))`,

/* ---------- Text (2) ---------- */
`Dear user,

This is plain prose that mentions angle brackets like <not-code> and braces {just text}.
It also writes 'int main' and 'SELECT *' but it's documentation, not executable code.

Thanks.`,

`From: "Alice Example" <alice@example.com>
To: Bob <bob@example.com>
Subject: Meeting notes (no attachments)
Date: Fri, 01 Aug 2025 09:00:00 +0200

- We discussed timelines.
- Next steps: follow up by Monday.`
];


document.addEventListener('DOMContentLoaded', bootstrap);
