// Exercise the generated visual-filter report with offline DOM/media stubs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag) {
    this.tagName = tag; this.children = []; this.style = {}; this.listeners = {};
    this.attributes = {}; this.textContent = ''; this.currentTime = 3.5;
    this.paused = true; this.readyState = 1; this.value = '';
    this.classList = {add() {}, remove() {}, toggle() {}};
  }
  append(...items) {this.children.push(...items);}
  appendChild(item) {this.children.push(item);}
  replaceChildren(...items) {this.children = items;}
  setAttribute(key,value) {this.attributes[key] = value;}
  getAttribute(key) {return this.attributes[key];}
  addEventListener(event,handler) {(this.listeners[event] ||= []).push(handler);}
  getBoundingClientRect() {return {width: 1000, height: 190, left: 0};}
  getContext() {return new Proxy({}, {get: (_,key) => key === 'measureText' ? () => ({width:100}) : () => {}});}
  scrollIntoView() {}
  pause() {this.paused = true;}
  load() {}
}
const elements = new Map();
const get = id => {if(!elements.has(id)) elements.set(id,new Element(id)); return elements.get(id);};
const source = fs.readFileSync(process.argv[2],'utf8');
const context = vm.createContext({document:{getElementById:get,createElement:tag=>new Element(tag),
  createElementNS:(_,tag)=>new Element(tag),querySelectorAll:()=>get('case-table').children},
  window:{devicePixelRatio:1,addEventListener() {},requestAnimationFrame() {}}});
vm.runInContext(source.match(/<script>([\s\S]*?)<\/script>/)[1],context);
if(process.argv[3] === 'real-active') {
  assert.equal(get('visual-panel').hidden,false);
  const decisions = vm.runInContext('REPORT.visualFilter.decisions',context);
  for(const item of decisions.filter(item=>item.metrics)) {
    get('video').currentTime=(item.source_start_seconds+item.source_end_seconds)/2;
    vm.runInContext('render()',context);
    assert.match(get('candidate-title').textContent,new RegExp(item.status === 'candidate'
      ? 'CONSERVAR CON FILTRO VISUAL' : item.status === 'rejected'
      ? 'DESCARTAR POR FILTRO VISUAL' : 'REVISAR EVIDENCIA VISUAL'));
  }
} else if(process.argv[3] === 'real-benchmark') {
  const cases = vm.runInContext('REPORT.cases',context);
  for(let index=0; index<cases.length; index++) {
    vm.runInContext(`chooseCase(${index})`,context);
    assert.equal(get('timeline').attributes.viewBox,'0 0 1000 248');
    for(const item of cases[index].visual_filter.decisions) {
      const t=(item.source_start_seconds+item.source_end_seconds)/2;
      vm.runInContext(`update(${t})`,context);
      assert.equal(get('library-state').textContent,item.status === 'candidate' ? 'Conservar' : 'Descartar');
    }
  }
} else if(process.argv[3] === 'benchmark') {
  vm.runInContext('update(3.5)',context);
  assert.match(get('evidence').children.map(item=>item.textContent).join(' '),/Giro lateral/);
  assert.match(get('visual-comparison').textContent,/Antes del filtro visual/);
  assert.equal(get('timeline').attributes.viewBox,'0 0 1000 248');
  assert.ok(get('timeline').children.some(item=>item.textContent === 'Antes visual'));
  assert.equal(get('library-state').textContent,'Descartar');
  vm.runInContext('update(4.5)',context);
  assert.equal(get('library-state').textContent,'Conservar');
} else {
  assert.equal(get('visual-panel').hidden,false);
  assert.match(get('visual-note').textContent,/Antes: 8.00s; después: 7.00s/);
  assert.match(get('candidate-title').textContent,/DESCARTAR POR FILTRO VISUAL/);
  const decisions = vm.runInContext('finalDecisions().filter(item=>item.kind === "run")',context);
  assert.equal(decisions.length,3);
  assert.equal(decisions[1].decision,'discard');
  get('video').currentTime=4.5;
  vm.runInContext('render()',context);
  assert.match(get('candidate-title').textContent,/CONSERVAR CON FILTRO VISUAL/);
  get('visual-decisions').children[1].listeners.click[0]();
  assert.equal(get('video').currentTime,3);
}
console.log('Visual filter UI interactions passed');
