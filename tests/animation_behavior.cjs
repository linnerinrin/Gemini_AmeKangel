const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const timers = new Map(); let id = 0;
const el = {src: '', closest: () => null};
const cfg = {frameSource: () => ({idle:['idle.png'], think:['t1.png','t2.png'], smile:['smile.png'], talk:['t1.png','t2.png']}), path:'/', elId:'test', loopActions:['idle'], getSuffix:()=>'', getIdle:()=> 'idle'};
const ctx = {BASE:'', NGO:{_statHandlers:[]}, charConfigs:{ame:cfg,kangel:cfg,dangel:cfg,'ame-live':cfg}, document:{getElementById:()=>el}, setInterval(fn){timers.set(++id,fn);return id;}, clearInterval(n){timers.delete(n);},setTimeout(){return ++id;},clearTimeout(){}, Math};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/js/service/animation.js'),'utf8'),ctx);
const anim = ctx.createCharAnim(cfg);
const tick = () => [...timers.values()].forEach(fn=>fn());
anim.setThinking(true);
for(let n=0;n<10;n++){tick();assert.match(el.src,/t[12]\.png$/);}
anim.playAction('smile'); assert.match(el.src,/t[12]\.png$/);
anim.setThinking(false); assert.equal(el.src,'/idle.png');
anim.playAction('smile');tick();assert.equal(el.src,'/idle.png');
anim.playAction('think',{loop:true});for(let n=0;n<8;n++){tick();assert.match(el.src,/t[12]\.png$/);}
// 状态轮询算出同一组 idle 帧时不应重启播放（否则每次轮询都抽一下）
anim.setThinking(false);
el.src = '/sentinel.png';
anim.playAction('idle');
assert.equal(el.src, '/sentinel.png', 'same idle key must not restart playback');
tick();
assert.equal(el.src, '/idle.png', 'idle timer still alive after skip');
anim.stop();assert.equal(timers.size,0);
console.log('Animation behavior passed: sustained thinking, click guard, release and transient recovery');
