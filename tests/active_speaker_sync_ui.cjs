// Execute actual generated JavaScript with offline canvas/media stubs.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor() {
    this.children = []; this.style = {}; this.listeners = {};
    this.textContent = ''; this.currentTime = 0.7; this.paused = true;
    this.classList = {add() {}, remove() {}};
  }
  append(...items) {this.children.push(...items);}
  appendChild(item) {this.children.push(item);}
  replaceChildren(...items) {this.children = items;}
  addEventListener(event, handler) {(this.listeners[event] ||= []).push(handler);}
  getBoundingClientRect() {return {width: 1000, height: 190, left: 0};}
  getContext() {return new Proxy({}, {get: (_, key) => key === 'measureText' ? () => ({width: 100}) : () => {}});}
  scrollIntoView() {}
}
const elements = new Map();
const get = id => {if(!elements.has(id)) elements.set(id,new Element()); return elements.get(id);};
const source = fs.readFileSync(process.argv[2], 'utf8');
const context = vm.createContext({document:{getElementById:get,createElement:()=>new Element()},
  window:{devicePixelRatio:1,addEventListener() {},requestAnimationFrame() {}}});
vm.runInContext(source.match(/<script>([\s\S]*?)<\/script>/)[1],context);
assert.equal(get('sync-panel').hidden, false);
assert.equal(get('sync-windows').children.length, 1);
assert.match(get('candidate-title').textContent, new RegExp(process.argv[3]));
const decisions = vm.runInContext('finalDecisions()',context);
const run = decisions.find(item=>item.kind==='run');
assert.equal(run.decision,process.argv[4]);
assert.ok(run.reasons.includes(process.argv[5]));
get('sync-windows').children[0].listeners.click[0]();
assert.equal(get('video').currentTime,0.6);
console.log('Synchronization UI interactions passed');
