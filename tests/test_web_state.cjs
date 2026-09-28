// Node-only state regression tests. This fake DOM is not a browser/visual test.
// Run: node tests/test_web_state.cjs
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(require('node:path').join(__dirname, '..', 'webui.py'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const flush = () => new Promise(resolve => setImmediate(resolve));

function setup(saved = new Map()) {
  const elements = new Map(), downloads = [], requests = [], revoked = [];
  class Element {
    constructor(tag = 'div') {
      this.tag = tag; this.children = []; this.dataset = {}; this.attrs = {};
      this.style = {setProperty() {}}; this.classList = {add() {}, remove() {}, toggle() {}};
      this.value = ''; this.checked = false; this.type = ''; this.textContent = '';
      this.previousElementSibling = {id: 'caption'};
      this.naturalWidth = 64; this.naturalHeight = 64;
      this.clientWidth = 800; this.clientHeight = 600;
    }
    setAttribute(k, v) { this.attrs[k] = v; }
    getAttribute(k) { return this.attrs[k]; }
    removeAttribute(k) { delete this.attrs[k]; }
    append(...nodes) { this.children.push(...nodes); }
    add(node) { this.append(node); }
    replaceChildren(...nodes) { this.children = nodes; }
    querySelectorAll(selector) { return selector === 'button' ? this.children.filter(x => x.tag === 'button') : []; }
    addEventListener() {}
    remove() {}
    click() { if (this.tag === 'a') downloads.push(this.download); this.onclick?.({stopPropagation() {}}); }
  }
  for (const match of html.split('<script>')[0].matchAll(/<([a-z]+)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const el = new Element(match[1]); el.id = match[3];
    for (const a of match[2].matchAll(/([\w-]+)="([^"]*)"/g)) el[a[1]] = a[2];
    el.checked = /\bchecked\b/.test(match[2]); elements.set(el.id, el);
  }
  for (const [id, values, key] of [['srMode',['off','1x','2x'],'sr'],['seg',['1','0'],'fit']]) {
    for (const value of values) { const el = new Element('button'); el.dataset[key] = value; elements.get(id).append(el); }
  }
  const doc = {
    documentElement: {dataset: {}}, body: new Element('body'), title: '',
    querySelector(selector) { if (selector === '.tag.r') return new Element(); return elements.get(selector.slice(1)); },
    querySelectorAll() { return []; }, createElement: tag => new Element(tag), addEventListener() {}
  };
  let urlId = 0;
  const context = vm.createContext({
    document: doc, navigator: {language: 'ru'}, window: {addEventListener() {}},
    localStorage: {getItem: key => saved.get(key) ?? null, setItem: (key,value) => saved.set(key,value)},
    matchMedia: () => ({matches: false}), getComputedStyle: () => ({paddingLeft:'20',paddingRight:'20',paddingTop:'20',paddingBottom:'20'}),
    ResizeObserver: class {observe() {}}, Option: class extends Element {constructor(text,value){super('option');this.textContent=text;this.value=value;}},
    URL: {createObjectURL: () => 'blob:'+ ++urlId, revokeObjectURL: url => revoked.push(url)},
    Image: class {set src(value){queueMicrotask(() => this.onload?.());}},
    FileReader: class {readAsDataURL(file){this.result='data:image/png;base64,eA==';queueMicrotask(() => this.onload());}},
    Blob, Uint8Array, atob, AbortController, setTimeout, clearTimeout,
    confirm: () => true,
    fetch: (url, options) => new Promise(resolve => requests.push({body:JSON.parse(options.body),resolve})),
  });
  vm.runInContext(script, context);
  return {context,elements,requests,downloads,revoked,saved,run: code => vm.runInContext(code,context)};
}
const files = '[{name:"one.png",size:100,type:"image/png"},{name:"two.png",size:100,type:"image/png"}]';
const reply = request => request.resolve({ok:true,json:async()=>({result:'data:image/png;base64,eA==',w:64,h:64,ms:1})});

(async () => {
  const app = setup(new Map([['stopGPTslop.lang','invalid']]));
  app.run('addFiles('+files+')');
  app.elements.get('a').value = '.75'; app.elements.get('a').oninput();
  assert.equal(app.requests.length, 0, 'file selection and sliders must not generate');
  app.elements.get('presetName').value = '<b>My preset</b>';
  app.elements.get('savePreset').onclick();
  assert.equal(app.requests.length, 0, 'saving must not generate');
  const reloaded = setup(app.saved);
  assert.equal(reloaded.run('savedPresets.length'), 1);
  reloaded.elements.get('userPresets').value='0';reloaded.elements.get('userPresets').onchange();
  assert.equal(reloaded.run('settings().alpha'), .75);

  const pending = app.run('runQueue()'); await flush();
  assert.equal(app.requests.length, 1);
  app.elements.get('a').value = '1.5'; app.elements.get('a').oninput();
  app.run('selectItem(2)'); reply(app.requests[0]); await flush();
  assert.equal(app.requests.length, 2); assert.equal(app.requests[1].body.alpha, .75, 'batch snapshots settings');
  assert.equal(app.run('current().result'), null, 'old response must not attach to another image');
  reply(app.requests[1]); await pending;
  app.elements.get('dl').onclick();
  assert.match(app.downloads[0], /two_clean_a0\.75_/, 'download uses completed settings');
  assert.equal(app.run('dirty(current())'), true);
  app.run('applySettings(savedPresets[0].settings)');
  await app.run('runQueue()'); assert.equal(app.requests.length,2,'identical results are skipped');

  const cleared=setup();cleared.run('addFiles('+files+')');
  const work=cleared.run('runQueue()');await flush();
  cleared.elements.get('reset').onclick();
  cleared.run('addFiles([{name:"new.png",size:20,type:"image/png"}])');
  reply(cleared.requests[0]);await work;
  assert.equal(cleared.run('queue.length'),1);assert.equal(cleared.run('current().result'),null);
  assert.equal(cleared.requests.length,1,'clear cancels remaining batch');

  const stopped=setup();stopped.run('addFiles('+files+')');
  const stopping=stopped.run('runQueue()');await flush();stopped.elements.get('stop').onclick();
  reply(stopped.requests[0]);await stopping;
  assert.equal(stopped.requests.length,1);assert.equal(stopped.run('queue[0].state'),'ready');
  const resume=stopped.run('runQueue()');await flush();reply(stopped.requests[1]);await resume;
  assert.equal(stopped.run('queue[1].state'),'ready');

  const failed=setup();failed.run('addFiles('+files+')');
  const failing=failed.run('runQueue()');await flush();
  failed.requests[0].resolve({ok:false,json:async()=>({error:'bad input'})});await flush();
  reply(failed.requests[1]);await failing;
  assert.equal(failed.run('queue[0].state'),'error');assert.equal(failed.run('queue[1].state'),'ready');
  const retry=failed.run('runQueue()');await flush();reply(failed.requests[2]);await retry;
  assert.equal(failed.run('queue[0].state'),'ready');
  console.log('PASS: manual generation, immutable batch, selection race, clear race, stop/resume, failures/retry, presets/reload, download settings, skip completed');
})().catch(error=>{console.error(error);process.exitCode=1;});
