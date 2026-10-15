const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname, '..');
function harness() {
  const elements = new Map(), listeners = new Set(), timers = new Set(), audios = [];
  function el(id) {
    if (!elements.has(id)) elements.set(id, {style:{}, textContent:'', checked:true, handlers:{}, classList:{add(){},remove(){},toggle(){},contains(){return false;}}, addEventListener(type,fn){this.handlers[type]=fn;}});
    return elements.get(id);
  }
  class Audio {
    constructor(src) { this.src=src; this.currentTime=0; this.duration=100; this.calls=[]; this.pauses=0; audios.push(this); }
    play() { return new Promise((resolve,reject)=>this.calls.push({resolve,reject})); }
    pause() { this.pauses++; }
    addEventListener() {}
  }
  const c = vm.createContext({Audio, Promise, BASE:'/static', NGO:{}, document:{getElementById:el,querySelectorAll:()=>[],querySelector:()=>null,addEventListener:(t,f)=>listeners.add(f),removeEventListener:(t,f)=>listeners.delete(f)}, setInterval(fn){const id={fn};timers.add(id);return id;},clearInterval:id=>timers.delete(id), startAmeAnimation(){},loadSaveSlots(){},playSfx(){},saveSelected:true});
  for (const file of ['service/bgm.js','ui/bgm-menu.js','service/desktop-session.js']) vm.runInContext(fs.readFileSync(path.join(root,'static/js',file),'utf8'), c);
  return {c,el,audios,listeners,timers};
}
const flush = async()=>{await Promise.resolve();await Promise.resolve();};
(async()=>{
  let h=harness(), c=h.c;
  c.playBGM('one'); let a=h.audios[0]; a.calls[0].resolve(); await flush();
  a.currentTime=37; c.stopBGM();
  assert.equal(c.bgmAudio,a); assert.equal(a.currentTime,37); assert.equal(h.el('psyche-time').textContent,'0:37 / 1:40'); assert.equal(h.el('psyche-bar-fill').style.width,'37%'); assert.equal(h.timers.size,0);
  c.playBGM('one'); assert.equal(h.audios.length,1); assert.equal(a.currentTime,37); a.calls[1].resolve(); await flush(); assert.equal(c.bgmPlaying,true);
  c.playBGM('two'); assert.equal(h.audios.length,2); assert.equal(c.bgmAudio.currentTime,0); assert.equal(h.el('psyche-bar-fill').style.width,'0%');
  console.log('PASS pause progress, same-track resume, new-track reset');
  h=harness(); c=h.c; c.playBGM('one'); a=h.audios[0]; a.calls[0].reject({name:'NotAllowedError'}); await flush();
  assert.equal(h.listeners.size,2); const pending=c._bgmPendingResume; c.stopBGM(); assert.equal(c._bgmPendingResume,null); assert.equal(h.listeners.has(pending),false); pending(); assert.equal(a.calls.length,1);
  c.revealDesktop(); assert.equal(a.calls.length,1); assert.equal(h.audios.length,1);
  console.log('PASS explicit pause cancels autoplay listener and desktop reveal preserves pause');
  h=harness(); c=h.c; c.playBGM('one'); a=h.audios[0]; c.stopBGM(); a.calls[0].resolve(); await flush(); assert.equal(c.bgmPlaying,false); assert.equal(h.timers.size,0);
  c.playBGM('one'); c.playBGM('two'); let b=h.audios[1]; b.calls[0].resolve(); await flush(); a.calls[1].reject({name:'NotAllowedError'}); await flush(); assert.equal(c.bgmPlaying,true); assert.equal(c.currentBGM,'two'); assert.equal(c._bgmPendingResume,null);
  console.log('PASS stale resolve/reject cannot revive paused or superseded playback');
  h=harness(); c=h.c; c.playBGM('one'); a=h.audios[0]; c.stopBGM(); c.playBGM('one'); a.calls[1].resolve(); await flush(); let pauses=a.pauses; a.calls[0].resolve(); await flush(); assert.equal(a.pauses,pauses); assert.equal(c.bgmPlaying,true);
  console.log('PASS stale same-Audio resolve does not interrupt newer resume');
  h=harness(); c=h.c; c.playBGM('one'); a=h.audios[0]; h.el('psyche-capsule-btn').handlers.click({stopPropagation(){}}); assert.equal(c.bgmWantsPlayback,false); a.calls[0].reject({name:'NotAllowedError'}); await flush(); assert.equal(c._bgmPendingResume,null);
  console.log('PASS capsule pauses pending playback before play settles');
  h=harness(); c=h.c; c.playBGM('one'); a=h.audios[0]; a.calls[0].reject({name:'NotAllowedError'}); await flush(); let resume=c._bgmPendingResume; resume(); assert.equal(a.calls.length,2); assert.equal(h.listeners.has(resume),false); a.calls[1].resolve(); await flush(); assert.equal(c.bgmPlaying,true); assert.equal(h.timers.size,1);
  console.log('PASS allowed user gesture retries blocked autoplay exactly once');
})().catch(err=>{console.error(err);process.exitCode=1;});
