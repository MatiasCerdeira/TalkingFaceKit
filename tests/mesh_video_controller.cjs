// Browser-free regression tests: run the actual exported controller with a simulated media clock.
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const html = fs.readFileSync(process.argv[2], "utf8");
const data = JSON.parse(html.match(/<script id="mesh-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const controller = html.match(/<script id="mesh-controller">([\s\S]*?)<\/script>/)[1];
const callbacks = process.argv[3] === "True";
class Element {
  constructor() { this.listeners = new Map(); this.textContent = ""; this.value = "0"; this.disabled = true; }
  addEventListener(name, handler) { this.listeners.set(name, handler); }
  emit(name) { return this.listeners.get(name)?.({target:this}); }
}
const nodes = new Map();
for (const name of ["mesh-data","video","mesh","status","clock","error","timeline","previous","next","play","range","local-video"]) nodes.set(name, new Element());
nodes.get("mesh-data").textContent = JSON.stringify(data);
const video = nodes.get("video");
Object.assign(video, {readyState:1, duration:100, currentTime:0, paused:true,
  pause() { this.paused = true; this.emit("pause"); },
  async play() { this.paused = false; this.emit("play"); },
  load() { this.emit("loadedmetadata"); },
});
if (callbacks) video.requestVideoFrameCallback = callback => { video.onFrame = callback; };
const updates = [];
let blockUpdates = false;
const pendingUpdates = [];
vm.runInNewContext(controller, {
  document:{getElementById:id => nodes.get(id)},
  Plotly:{newPlot:async () => {}, restyle:async (_plot, update) => {
    updates.push(update);
    if (blockUpdates) await new Promise(resolve => pendingUpdates.push(resolve));
  }},
  atob, Uint8Array, Float32Array, DataView, requestAnimationFrame:() => {},
  URL:{createObjectURL:() => "blob:local", revokeObjectURL:() => {}},
});
const flush = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
function present(sourceTime) {
  if (callbacks) video.onFrame(0, {mediaTime:sourceTime - data.videoOffset});
  else { video.currentTime = sourceTime - data.videoOffset; video.emit("timeupdate"); }
}
(async () => {
  await flush();
  assert.equal(video.currentTime, 20 - data.videoOffset);
  assert.equal(nodes.get("timeline").disabled, false);
  video.emit("seeked"); await flush();
  assert.equal(updates.at(-1).visible[0], true);
  assert.match(nodes.get("status").textContent, /Frame 600/);

  // With frame callbacks, deliberately disagree with currentTime: mediaTime must win.
  video.paused = false; video.currentTime = 20.11 - data.videoOffset;
  present(20.03); await flush();
  assert.equal(updates.at(-1).visible[0], false);
  assert.match(nodes.get("status").textContent, /Frame 601.*Sin estimación/);
  present(20.11); await flush();
  assert.equal(updates.at(-1).visible[0], true);
  assert.equal(updates.at(-1).x[0][0], 0.5);
  assert.match(nodes.get("status").textContent, /Frame 602/);

  // Irregular timestamp gaps are selected by binary search, not by a nominal FPS.
  present(20.08); await flush();
  assert.equal(updates.at(-1).visible[0], false);
  assert.match(nodes.get("status").textContent, /Frame 601/);

  video.currentTime = 20.11 - data.videoOffset;
  nodes.get("previous").emit("click");
  assert.equal(video.paused, true);
  assert.equal(video.currentTime, 20.03 - data.videoOffset);
  video.emit("seeked"); await flush();
  assert.match(nodes.get("status").textContent, /Frame 601/);
  nodes.get("timeline").value = "0";
  nodes.get("timeline").emit("input");
  assert.equal(video.currentTime, 20 - data.videoOffset);
  video.emit("seeked"); await flush();
  assert.match(nodes.get("status").textContent, /Frame 600/);

  video.paused = false;
  present(21); await flush();
  assert.equal(video.paused, true);
  assert.equal(video.currentTime, 20.11 - data.videoOffset);
  video.emit("seeked"); await flush();
  assert.match(nodes.get("status").textContent, /Frame 602/);

  // A slow renderer must coalesce intermediate requests rather than accumulating stale frames.
  blockUpdates = true;
  present(20); present(20.03); present(20.11);
  assert.equal(pendingUpdates.length, 1);
  blockUpdates = false;
  pendingUpdates.shift()();
  await flush();
  assert.match(nodes.get("status").textContent, /Frame 602/);
  assert.equal(updates.at(-1).visible[0], true);
  assert.equal(updates.at(-1).x[0][0], 0.5);
  assert.equal(nodes.get("error").textContent, "");
})().catch(error => { console.error(error); process.exitCode = 1; });
