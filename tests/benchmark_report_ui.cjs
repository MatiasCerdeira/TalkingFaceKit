// Exercise the generated report's interaction logic offline, without browser or media access.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.style = {};
    this.textContent = '';
    this.value = '';
    this.currentTime = 0;
    this.readyState = 0;
    this.paused = true;
    this.duration = 100;
    this.classList = { toggle() {} };
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = [...children]; }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  addEventListener(event, callback) { (this.listeners[event] ||= []).push(callback); }
  async dispatch(event, fields = {}) {
    for (const callback of this.listeners[event] || []) await callback({ target: this, ...fields });
  }
  pause() { this.paused = true; }
  async play() { this.paused = false; }
  load() { this.readyState = 0; }
  set src(value) { this.attributes.src = value; }
  getBoundingClientRect() { return { left: 0, width: 1000 }; }
  scrollIntoView() {}
}

async function main() {
  const source = fs.readFileSync(process.argv[2], 'utf8');
  const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, new Element(id));
    return elements.get(id);
  };
  const context = vm.createContext({
    document: {
      getElementById: get,
      createElement: tag => new Element(tag),
      createElementNS: (_, tag) => new Element(tag),
      querySelectorAll: () => get('case-table').children,
    },
  });
  vm.runInContext(script, context);
  const video = get('video');
  const emptyBands = () => get('timeline').children.filter(child =>
    child.tagName === 'text' && child.textContent.startsWith('Sin intervalos conservados'));
  assert.equal(emptyBands().length, 0);
  assert.equal(get('differences').children.length, 4);
  // Selecting a difference before metadata loads preserves the requested source timestamp.
  await get('differences').children[0].dispatch('click');
  video.readyState = 1;
  await video.dispatch('loadedmetadata');
  assert.equal(video.currentTime, 31);
  assert.equal(get('human-state').textContent, 'Descartar');
  assert.equal(get('library-state').textContent, 'Conservar');
  assert.equal(get('view-label').textContent, 'Entorno de la diferencia · ±3 s');
  await get('next').dispatch('click');
  assert.equal(video.currentTime, 34);
  assert.equal(get('status').textContent, 'Útil perdido');
  assert.match(get('evidence').children.map(child => child.textContent).join(' '), /varios rostros a la vez/);
  await get('play-context').dispatch('click');
  assert.equal(video.currentTime, 32.5);
  assert.equal(video.paused, false);
  video.currentTime = 36.75;
  await video.dispatch('timeupdate');
  assert.equal(video.currentTime, 36.5);
  assert.equal(video.paused, true);
  get('filter').value = 'false_negative';
  await get('filter').dispatch('change');
  assert.equal(get('differences').children.length, 1);
  await get('next').dispatch('click');
  assert.equal(video.currentTime, 34);
  assert.equal(get('next').disabled, true);
  await get('reset-view').dispatch('click');
  assert.equal(get('view-label').textContent, 'Ventana completa');
  get('seek').value = '38.5';
  await get('seek').dispatch('input');
  assert.equal(video.currentTime, 38.5);
  assert.equal(get('human-state').textContent, 'Sin evaluar');
  assert.equal(get('status').textContent, 'Sin evaluar');
  // A click halfway across the actual plot uses source seconds, not clip-relative time.
  await get('timeline').dispatch('click', { clientX: 115 + 870 / 2 });
  assert.equal(video.currentTime, 35);
  get('case-select').value = '1';
  await get('case-select').dispatch('change');
  assert.equal(video.currentTime, 50);
  assert.equal(emptyBands().length, 2);
  assert.equal(get('human-state').textContent, 'Descartar');
  assert.equal(get('library-state').textContent, 'Descartar');
  assert.equal(get('status').textContent, 'Ambos descartan');
  assert.deepEqual(get('case-metrics').children.slice(0, 2).map(card => card.children[0].textContent), ['—', '—']);
  assert.match(get('differences').children[0].textContent, /No hay diferencias/);
  assert.equal(get('next').disabled, true);
  assert.equal(get('play-context').disabled, true);
  assert.match(get('evidence').children[0].textContent, /no guardó los motivos/);
  video.currentTime = 61;
  await video.dispatch('timeupdate');
  assert.equal(video.currentTime, 60);
  assert.equal(get('status').textContent, 'Fin de la ventana evaluada');
  assert.equal(get('human-state').textContent, '—');
  await video.dispatch('error');
  assert.equal(get('media-error').hidden, false);
  assert.match(get('media-error').textContent, /No se pudo reproducir/);
  // Empty-band captions must clear when switching back to a case with selections.
  get('case-select').value = '0';
  await get('case-select').dispatch('change');
  assert.equal(emptyBands().length, 0);
  console.log('Comparison UI interactions passed');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
