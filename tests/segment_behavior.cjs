// P4：聊天模式按换行分段发送（分段依据已从 @@@ 标记改回换行 \n）
// 用 tests/stream_fake_dom.cjs 的 El 与假 timers()。
// 分段闸门是"1 秒后再看一次"的真实定时器，所以测试给被测代码一个
// 由测试推进的虚拟时钟（vm 里的 Date.now 指向 vnow），step(ms) 精确步进、
// 只触发已经到点的定时器 —— step(999) 与 step(1000) 因此可以精确对照。
const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const {El,timers}=require('./stream_fake_dom.cjs');
const root=path.resolve(__dirname,'..');
const clock=timers();
const EPOCH=1700000000000;
// 被测代码的定时器全是假的，测试却全用 await：没有挂起的真实 I/O 时 Node 会
// 在微任务清空后直接退出（那会变成"断言没跑完就 exit 0"的假绿）。用真实心跳撑住事件循环。
const heartbeat=setInterval(function(){},50);
// 共享的 timers() 只按"相对 delay"判到点，没法配合推进虚拟时钟。
// 这里在测试内包一层：自己记录每个定时器的绝对到点时刻，然后按虚拟时间精确步进。
// （不改动 tests/stream_fake_dom.cjs，其它测试的语义完全不受影响。）
const at=new Map();   // timerId -> 绝对到点时刻
const rawSetTimeout=clock.setTimeout.bind(clock), rawClearTimeout=clock.clearTimeout.bind(clock);
let vnow=EPOCH;
clock.setTimeout=function(fn,delay){
  const id=rawSetTimeout(fn,delay);
  at.set(id,vnow+(Number(delay)||0));
  return id;
};
clock.clearTimeout=function(id){ at.delete(id); rawClearTimeout(id); };
// 触发所有已到点的定时器
function fireDue(){
  for(const [id,job] of [...clock.jobs]){
    if(at.get(id)<=vnow){ clock.jobs.delete(id); at.delete(id); job.fn(); }
  }
}
// 精确步进 ms：按到点顺序逐个触发，最后停在 vnow+ms
function step(ms){
  const goal=vnow+ms;
  for(;;){
    let next=null;
    for(const id of clock.jobs.keys()){ const t=at.get(id); if(t!==undefined&&t<=goal&&(next===null||t<next))next=t; }
    if(next===null)break;
    vnow=next;
    fireDue();
  }
  vnow=goal;
}

const ids=new Map(), listeners=new Map();
function clearListeners(){listeners.clear();}
function el(id){
  var e=ids.get(id);
  if(!e){ e=new El(); Object.defineProperty(e,'_id',{value:id,enumerable:false}); ids.set(id,e); }
  return e;
}
var selected={isCollapsed:true,rangeCount:0,anchorNode:null},closeWindow=null;
const ctx=vm.createContext({
  console,
  // 被测代码只读 Date.now()：指向可控虚拟时钟
  Date:{now:function(){return vnow;}},
  showError:function(){},
  AbortController:AbortController,
  BASE:'/static',
  currentTopic:{id:0,mode:'1'},
  liveTimer:null,
  playSfx:function(){}, playJineAction:function(){}, playLiveAction:function(){},
  ameAnim:{setThinking:function(){}}, kangelAnim:{setThinking:function(){}},
  dangelAnim:{setThinking:function(){}}, ameLiveAnim:{setThinking:function(){}},
  setTimeout:clock.setTimeout, clearTimeout:clock.clearTimeout,
  setInterval:function(){return 0;}, clearInterval:function(){},
  requestAnimationFrame:function(){return 0;}, cancelAnimationFrame:function(){},
  getSelection:function(){return selected;},
  document:{
    // 统一返回替身元素：live.js 会直接给窗口/提示元素挂监听，返回 null 会炸在加载阶段
    getElementById:function(id){return el(id);},
    querySelector:function(s){
      s=String(s);
      if(s.charAt(0)==='#')return el(s.slice(1));
      if(s.charAt(0)==='.')return el(s);   // JINE 用 .jine-chat-area 这类选择器
      return null;
    },
    querySelectorAll:function(){return [];},
    createElement:function(t){return new El(t);},
    createTextNode:function(t){var n=new El('#text');n.textContent=t;return n;}
  }
});
ctx.window=ctx;
ctx.NGO={topicConfigs:[],WindowManager:{register:function(e,o){closeWindow=o&&o.onClose;}}};
function load(f){vm.runInContext(fs.readFileSync(path.join(root,f),'utf8'),ctx);}
load('static/js/data/registry.js');
load('static/js/service/text.js');
load('static/js/service/thinking.js');
load('static/js/service/chat.js');
load('static/js/ui/live.js');

const ngo=ctx.NGO,chat=ngo.liveChat,channel=ctx.liveChannel,area=el('live-danmaku'),overlay=el('live-reply-overlay'),input=el('live-chat-input');
// 只数机器人行（行里有 botIcon），用户消息行与空气泡都不算
const botRows=()=>area.childNodes.filter(function(n){
  return n.children.some(function(c){return c.src&&String(c.src).indexOf('icon_cho')!==-1;});
});
const rows=()=>botRows().length;
const items=()=>area.childNodes.length;
const texts=()=>area.childNodes.map(function(n){return n.textContent;});
// 气泡本体文本（不含思考块的标签/点号——思考块是同一行的兄弟节点）
const bubble=function(i){
  var n=area.childNodes[i];
  if(!n)return '';
  return n._contentTarget?n._contentTarget.textContent:n.textContent;
};
// 闸门定时器 = 段间隔定时器（恰好 1 秒量级）。
// 别把 live.js 的字幕自动隐藏定时器（>=5000ms）与思考点号（400ms）算进来。
const gateJobs=()=>[...clock.jobs.values()].filter(function(j){return j.delay>=900&&j.delay<5000;}).length;
const gateEta=function(){
  var due=[...clock.jobs.entries()].filter(function(e){return e[1].delay>=900&&e[1].delay<5000;}).map(function(e){return at.get(e[0]);});
  return due.length?Math.min.apply(null,due):null;
};

let life=[];
const realLife=channel.onAnswerLifecycle;
channel.onAnswerLifecycle=function(type,text,identity){life.push({type:type,text:text});return realLife.call(this,type,text,identity);};

const locked=function(){return !!ngo.chatBusy;};
function assertLocked(name){assert.ok(locked(),'输入锁应保持到最后一段出现（'+name+'）');}
function assertFree(name){assert.equal(locked(),false,'输入锁必须已释放（'+name+'）');}
function noGate(name){assert.equal(gateJobs(),0,'不该留下悬挂的分段闸门定时器（'+name+'）');}
const settleFlag=promise=>{const s={settled:false};promise.then(function(){s.settled=true;},function(){s.settled=true;});return s;};
const pushEvent=next=>{ngo.chatAPI={streamChat:next};};
const tick=async()=>{for(var i=0;i<4;i++)await Promise.resolve();};
// 发送一条消息：每次发送前把假输入框填回去（被测代码发送时会清空它）
function send(){
  input.value='hello';
  var p=chat.sendMessage();
  if(p&&p.catch)p.catch(function(){});
  return p;
}
// 用 registry.js 里 liveChannel 的同一套任务轴语义：读 NGO.taskMode，缺省兜底 chat。
// （不要硬编码 'chat'，否则 ngo.taskMode='work' 的用例会假绿/挂起。）
channel.getTaskMode=function(){return (typeof ngo.taskMode==='string'&&ngo.taskMode)?ngo.taskMode:'chat';};
function setTaskMode(mode){ if(mode)ngo.taskMode=mode; else delete ngo.taskMode; }
// 发起一条"done 一次性携带完整正文"的分段回复。
// 注意：这里同步返回 doSend 的 promise，绝不能 async 化——
// 否则 await segment(...) 会在调用方 step() 之前就把整个发送等完（自锁）。
function segment(ans){
  // ⚠️ 顺序要紧：clearMessages() 会因为上一轮还留着 _lastIdentity 补发一条 error 生命周期，
  // 必须在清空 life 之前调用，否则这条 error 会被算进本用例的生命周期断言。
  chat.clearMessages(); clearListeners(); life=[]; setTaskMode('chat');
  pushEvent(async function(p,m,tm,cb){cb({type:'done',answer:ans});});
  return send();
}
// JINE：真实 registry 里的 jineChannel，用来验证每段是独立「头像 + 气泡」
const jineArea=el('.jine-chat-area');
const jineRows=()=>jineArea.childNodes.filter(function(n){
  return n.children.some(function(c){return c.src&&String(c.src).indexOf('icon_jine_ame')!==-1;});
});
const jineTexts=()=>jineArea.childNodes.map(function(n){return n.textContent;});

(async()=>{
  // ── 1. 单行回复（整条没有换行）：与改造前完全一致，零变化回归安全网 ──
  // ⚠️ 顺序要紧：clearMessages() 会因为上一轮还留着 _lastIdentity 补发一条 error 生命周期，
  // 必须在清空 life 之前调用，否则这条 error 会被算进本用例的生命周期断言。
  chat.clearMessages(); clearListeners(); life=[]; setTaskMode('chat');
  const single='你说好，我就懒得搭理了。';
  let release1=null;
  pushEvent(async function(p,m,tm,cb){
    cb({type:'answer_delta',text:'你说好，我就'});
    await new Promise(function(r){release1=r;});   // 让 16ms 合并重绘先跑一次
    cb({type:'answer_delta',text:'懒得搭理了。'});
    cb({type:'done',answer:single});
  });
  const p1=send(); await tick(); step(16);
  const row1=area.childNodes[1];
  assert.equal(rows(),1,'单行回复只有一条机器人行');
  assert.equal(bubble(1),'你说好，我就','单行回复走原有整条渲染路径');
  assert.equal(gateJobs(),0,'单行回复绝不排定闸门定时器');
  release1(); await tick(); await p1;
  assert.equal(area.childNodes[1],row1,'单行回复原地增长：始终是同一个行节点（零变化）');
  assert.equal(items(),2,'用户 1 条 + 回复 1 条，没有多余空气泡');
  assert.equal(bubble(1),single);
  assert.deepEqual(life.map(function(e){return e.type;}),['start','update','done'],'未分段：重绘 update + 收尾 done');
  assert.equal(life[life.length-1].text,single,'收尾 done 带完整正文');
  assertFree('单行回复结束');
  noGate('单行回复结束');
  console.log('[case1] 单行回复：rows='+rows()+'，同一行节点原地增长='+(area.childNodes[1]===row1)+'，闸门定时器='+gateJobs()+'，life='+life.map(function(e){return e.type;}).join('/'));

  // ── 2. A\nB\nC → 3 条独立「头像 + 气泡」消息 ──
  // ⚠️ 顺序要紧：clearMessages() 会因为上一轮还留着 _lastIdentity 补发一条 error 生命周期，
  // 必须在清空 life 之前调用，否则这条 error 会被算进本用例的生命周期断言。
  chat.clearMessages(); clearListeners(); life=[]; setTaskMode('chat');
  pushEvent(async function(p,m,tm,cb){
    cb({type:'answer_delta',text:'A\nB\nC'});
    cb({type:'done',answer:'A\nB\nC'});
  });
  const p2=send(); await tick();
  assert.equal(rows(),1,'换行即封口：第一段立刻出现');
  assertLocked('只出现第一段时');
  step(1000);
  assert.equal(rows(),2,'第二段 1 秒后出现');
  assertLocked('第二段出现后');
  step(1000);
  assert.equal(rows(),3,'第三段再 1 秒后出现');
  await p2;
  assert.equal(items(),4,'用户 1 行 + 回复 3 段，没有多余空气泡');
  assert.deepEqual(texts(),['hello','A','B','C']);
  botRows().forEach(function(r,i){
    assert.equal(r.children.length,2,'每条机器人行是「头像 + 内容」两项，没有多余空元素');
    assert.ok(String(r.children[0].src).indexOf('icon_cho')!==-1,'第一项是头像');
    assert.equal(r._contentTarget.textContent,['A','B','C'][i],'每段一条独立消息');
  });
  assert.deepEqual(life.map(function(e){return e.type;}),['start','update','update','done'],'每段一次 update，最后一段收尾 done');
  assert.deepEqual(life.filter(function(e){return e.type==='update';}).map(function(e){return e.text;}),['A','B']);
  assert.equal(life[life.length-1].text,'C');
  assertFree('三段显示完');
  noGate('三段结束');

  // ── 3. 段间隔 ≥1 秒：999ms 时下一段未出现，1000ms 后出现 ──
  const p3=segment('S1\nS2');
  await tick();
  assert.equal(rows(),1,'第一段已出现');
  assert.equal(gateJobs(),1,'第一段之后必须已排定闸门定时器（不变式）');
  const eta=gateEta();
  const gap=eta-vnow;
  const t0=vnow;                     // 第一段出现时的虚拟时刻
  assert.ok(gap>900&&gap<=1000,'段间隔定时器应在 1 秒量级（实际 '+gap+'ms）');
  step(999);
  const before=eta-vnow;
  assert.ok(before>0,'校准点应仍在第二段到点之前（差 '+before+'ms）');
  assert.equal(rows(),1,'step(999) 时第二段不得出现');
  assert.equal(bubble(1).indexOf('S2'),-1,'step(999) 时第二段内容不得混进第一段');
  assertLocked('第二段出现前');
  assert.ok(gateJobs()>=1,'闸门关闭期间仍必须有已排定的定时器');
  console.log('[case3] step(999)：距到点还剩 ' + before + 'ms → rows=' + rows() + '，闸门定时器=' + gateJobs() + '，chatBusy 仍占用=' + locked() + '，第二段文本出现=' + (bubble(1).indexOf('S2')!==-1));
  step(1);
  assert.equal(rows(),2,'step(1000) 后第二段应出现');
  assert.deepEqual(texts(),['hello','S1','S2'],'第二段是新的独立气泡，第一段内容不被改写');
  await p3;
  assert.ok(vnow>=eta,'段间隔必须 ≥1 秒（虚拟时钟 '+vnow+' 应 ≥ 到点时刻 '+eta+'）');
  console.log('[case3] step(1000)：距第一段出现已过 ' + (vnow-t0) + 'ms（要求 >=1000）→ rows=' + rows() + '，文本=' + JSON.stringify(texts()));
  assertFree('第二段出现后');

  // ── 4. 输入锁：最后一段出现前 chatBusy 占用、sendMessage 未 settle；之后才释放 ──
  // ⚠️ 顺序要紧：clearMessages() 会因为上一轮还留着 _lastIdentity 补发一条 error 生命周期，
  // 必须在清空 life 之前调用，否则这条 error 会被算进本用例的生命周期断言。
  chat.clearMessages(); clearListeners(); life=[]; setTaskMode('chat');
  pushEvent(async function(p,m,tm,cb){cb({type:'done',answer:'L1\nL2\nL3'});});
  const p4=send(); const st4=settleFlag(p4); await tick();
  assertLocked('刚发送');
  assert.equal(rows(),1);
  await tick();
  assert.equal(st4.settled,false,'最后一段出现前 sendMessage() 不得 settle');
  assertLocked('只显示第一段');
  step(1000);
  assert.equal(rows(),2);
  await tick();
  assert.equal(st4.settled,false,'还有未显示的段，promise 不得 settle');
  assertLocked('显示到第二段');
  step(1000);
  assert.equal(rows(),3);
  await tick();
  assert.equal(st4.settled,true,'最后一段出现后 promise 应 settle');
  await p4;
  assertFree('最后一段出现后');
  noGate('三段结束');

  // ── 5. 历史回放：按换行分段、同步铺开、无延时（不留任何闸门定时器）──
  // 先 clearMessages 一次把 _lastIdentity 置空：loadHistory 内部还会再 clear 一次，
  // 否则那次 clear 会因为上一个请求还在而发一条 error（与分段无关）。
  clearListeners(); chat.clearMessages(); life=[];
  chat.loadHistory([
    {user:'q1',assistant:'H1\nH2\nH3',task_mode:'chat'},
    {user:'q2',assistant:'W1\nW2',task_mode:'work'},
    {user:'q3',assistant:'N1\n\n\nN2',task_mode:'chat'},
    {user:'q4',assistant:'单行回复',task_mode:'chat'}
  ],{timeBars:false});
  // 紧跟 loadHistory 同步断言：不需要 step 任何时钟 → 历史回放没有延时
  assert.deepEqual(texts(),['q1','H1','H2','H3','q2','W1\nW2','q3','N1','N2','q4','单行回复']);
  assert.equal(rows(),7,'历史：聊天回复按换行拆开，工作模式整条，单行整条');
  assert.equal(gateJobs(),0,'历史回放不得排定段间隔定时器：不留延时直接铺开');
  assert.equal(life.length,0,'历史回放保持 silent，不发字幕生命周期');

  // ── 6. 工作模式不分段：含 \n 的回复是一条气泡，换行原样保留 ──
  const workText='A\nB\nC';
  clearListeners(); chat.clearMessages(); life=[]; ngo.taskMode='work';
  pushEvent(async function(p,m,tm,cb){assert.equal(tm,'work');cb({type:'done',answer:workText});});
  const p6=send(); await tick(); step(1000); await p6;
  assert.equal(rows(),1,'工作模式整条一个气泡');
  assert.equal(bubble(1),workText,'工作模式的换行原样保留在同一条消息里');
  assert.equal(items(),2,'工作模式不产生空气泡');
  assert.equal(gateJobs(),0,'工作模式没有闸门定时器');
  delete ngo.taskMode;
  clearListeners(); chat.clearMessages(); life=[]; setTaskMode('chat');
  pushEvent(async function(p,m,tm,cb){cb({type:'done',answer:workText});});
  const p6c=send(); await tick();
  assert.equal(rows(),1,'聊天模式第一段先出现');
  step(1000);
  assert.equal(rows(),2,'同一文本在聊天模式必须按换行分段');
  step(1000);
  assert.equal(rows(),3);
  await p6c;
  assert.deepEqual(texts(),['hello','A','B','C']);
  assertFree('聊天模式分段结束');

  // ── 7. 空段丢弃：连续空行 / 只含空白的行不产生空气泡；\r\n 兼容 ──
  clearListeners(); chat.clearMessages(); life=[];
  chat.loadHistory([
    {user:'u1',assistant:'A\n\n\nB',task_mode:'chat'},
    {user:'u2',assistant:'C\n   \n\t\nD',task_mode:'chat'},
    {user:'u3',assistant:'E\r\nF\r\nG',task_mode:'chat'},
    {user:'u4',assistant:'\n\n  \n',task_mode:'chat'}
  ],{timeBars:false});
  assert.deepEqual(texts(),['u1','A','B','u2','C','D','u3','E','F','G','u4'],'空段被丢弃、\r\n 与 \n 同样分段');
  assert.equal(texts().join('').indexOf('\r'),-1,'段内容不残留 \\r');
  assert.equal(gateJobs(),0);

  // ── 8. 段内容不 trim：行首空格、缩进、尾部空格都原样保留 ──
  clearListeners(); chat.clearMessages();
  chat.loadHistory([{user:'t',assistant:'  保留行首\n\t缩进\t\n尾部  ',task_mode:'chat'}],{timeBars:false});
  assert.equal(rows(),3,'三个段');
  assert.equal(bubble(1),'  保留行首','段内容不得 trim（行首空格）');
  assert.equal(bubble(2),'\t缩进\t','段内容不得 trim（缩进）');
  assert.equal(bubble(3),'尾部  ','段内容不得 trim（尾部空格）');

  // ── 9. 代码围栏保护：围栏块不切段 ──
  // (a) 完整围栏：前言 + 围栏块 + 后记 = 3 段（块内的换行不切）
  clearListeners(); chat.clearMessages();
  chat.loadHistory([{user:'f',assistant:'前言\n```python\nif a:\n    b()\n```\n后记',task_mode:'chat'}],{timeBars:false});
  assert.deepEqual(texts(),['f','前言','```python\nif a:\n    b()\n```','后记'],'围栏块（含围栏行）算同一个段');
  // (b) 整条就是一个围栏块 → 1 段
  clearListeners(); chat.clearMessages();
  chat.loadHistory([{user:'f2',assistant:'```python\nif a:\n    b()\n```',task_mode:'chat'}],{timeBars:false});
  assert.deepEqual(texts(),['f2','```python\nif a:\n    b()\n```'],'整条围栏块不切段');
  // (c) 流式期间围栏还没闭合：按"仍在围栏内"处理，先不切
  clearListeners(); chat.clearMessages(); life=[]; setTaskMode('chat');
  const rel9=[];
  pushEvent(async function(p,m,tm,cb){
    cb({type:'answer_delta',text:'开头\n```python\nif a:'});   // 围栏未闭合
    await new Promise(function(r){rel9.push(r);});
    cb({type:'answer_delta',text:'\n    b()\n```'});           // 围栏闭合，块原地变长
    await new Promise(function(r){rel9.push(r);});
    cb({type:'done',answer:'开头\n```python\nif a:\n    b()\n```'});
  });
  const p9=send(); await tick();
  assert.equal(rows(),1,'第一段先出现');
  assert.equal(gateJobs(),1,'未闭合的围栏块是"增长中的末段"，必须已排定闸门定时器');
  step(1000);
  assert.equal(rows(),2,'未闭合围栏整块作为一段出现');
  assert.equal(bubble(2),'```python\nif a:','围栏内的换行没有被拆成多条消息');
  rel9[0](); await tick(); step(16);
  assert.equal(rows(),2,'围栏闭合后仍是同一段，不新增气泡');
  assert.equal(bubble(2),'```python\nif a:\n    b()\n```','围栏块原地增长，内部换行不拆段');
  assert.equal(bubble(1),'开头','已封口的段不得被改写');
  rel9[1](); await tick();
  await p9;
  assert.deepEqual(texts(),['hello','开头','```python\nif a:\n    b()\n```'],'最终仍是两条独立消息');
  assertFree('围栏保护结束');
  noGate('围栏保护结束');
  console.log('[case9] 围栏保护：整条围栏块 rows=' + rows() + '（块内 3 个换行未拆段），文本=' + JSON.stringify(texts()));

  // ── 10. 回归卡死①：首段经 16ms 合并重绘显示后，换行才到达（同一个流）──
  clearListeners(); chat.clearMessages(); life=[]; setTaskMode('chat');
  let release10=null;
  pushEvent(async function(p,m,tm,cb){
    cb({type:'answer_delta',text:'第一段'});
    cb({type:'answer_delta',text:'A'});            // 这两帧会被 16ms 合并成一次重绘
    await new Promise(function(r){release10=r;});  // 首段"第一段A"已经显示，换行此刻才到
    cb({type:'answer_delta',text:'\nB'});
    cb({type:'done',answer:'第一段A\nB'});
  });
  const p10=send(); await tick();
  step(16);
  assert.equal(rows(),1,'首段应经合并重绘显示');
  assert.equal(bubble(1),'第一段A','首段内容来自合并重绘');
  assert.equal(gateJobs(),0,'换行还没到，不该有闸门定时器');
  assertLocked('首段显示后');
  release10(); await tick();
  assert.equal(bubble(1),'第一段A','换行到达后首段封口，内容不变');
  assert.equal(gateJobs(),1,'换行到达后必须排定闸门定时器，否则后续段永不出现（上一版死锁①）');
  step(999);
  assert.equal([bubble(1),bubble(2)].every(function(t){return t!=='B';}),true,'不足 1 秒时补发段不得提前出现');
  step(1);
  await p10;
  assert.equal(rows(),2,'补发段最终必须出现，不得永久缺失');
  assert.deepEqual(texts(),['hello','第一段A','B'],'首段封口后补发段成为独立行');
  console.log('[regress1] 首段先经 16ms 合并重绘显示("第一段A")、换行后到 → 已排定闸门定时器，1 秒后补发段出现：rows=' + rows() + '，文本=' + JSON.stringify(texts()) + '，chatBusy 已释放=' + !locked());
  assertFree('补发段出现后');
  noGate('补发段结束');

  // ── 11. 回归卡死②：done 之后等剩余段期间 clearMessages() → 输入锁必须释放 ──
  clearListeners(); chat.clearMessages(); setTaskMode('chat');
  pushEvent(async function(p,m,tm,cb){cb({type:'done',answer:'X1\nX2'});});
  const p11=send(); const st11=settleFlag(p11); await tick();
  assert.equal(rows(),1);
  assertLocked('等待第二段时');
  const busyBeforeClear=locked();
  chat.clearMessages();          // 等剩余段期间清空会话 → cancelSeg 必须释放等待者
  await tick();
  assert.equal(st11.settled,true,'clearMessages 后 doSend 必须结束');
  await p11;
  assertFree('clearMessages 之后');
  noGate('取消后');
  assert.equal(rows(),0,'清空后消息区应为空');
  console.log('[regress2] done 后等剩余段期间 clearMessages()：doSend 已结束=' + st11.settled + '（清空前 chatBusy=' + busyBeforeClear + '）→ 释放后 chatBusy 占用=' + locked() + '，闸门定时器=' + gateJobs());

  // ── 12. 取消（abort）路径不得留下悬挂定时器，旧段不得再出现 ──
  const p12=segment('C1\nC2');
  await tick();
  assert.equal(rows(),1);
  assert.equal(gateJobs(),1);
  ngo.activeChatController.abort();
  await tick();
  await p12;
  assertFree('abort 后');
  noGate('abort 后');
  step(2000);
  assertFree('abort 后继续走时钟');
  assert.equal(rows(),0,'abort 后旧段不得再出现');

  // ── 13. 直播字幕逐段切换：每段一次 update（该段文本），收尾一次 done ──
  clearListeners(); life=[];
  const p13=segment('甲\n乙\n丙');
  await tick();
  assert.equal(rows(),1);
  step(1000); assert.equal(rows(),2);
  step(1000); assert.equal(rows(),3);
  await p13;
  assert.deepEqual(life.map(function(e){return e.type;}),['start','update','update','done']);
  assert.deepEqual(life.map(function(e){return e.text;}),['','甲','乙','丙']);
  assert.equal(overlay.textContent,'丙','字幕逐段切换，最后停在最后一段');
  assertFree('字幕逐段切换结束');
  noGate('字幕逐段切换结束');

  // ── 14. 不变式：任何时刻有未显示的段 ⇔ 存在已排定的闸门定时器（末段显示后为 0）──
  clearListeners(); chat.clearMessages(); setTaskMode('chat');
  let release14=null;
  pushEvent(async function(p,m,tm,cb){
    cb({type:'answer_delta',text:'I1\nI2\nI3'});
    await new Promise(function(r){release14=r;});
    cb({type:'done',answer:'I1\nI2\nI3'});
  });
  const p14=send(); await tick();
  assert.equal(rows(),1);
  assert.equal(gateJobs(),1,'有未显示的段却没有闸门定时器 → 就是上一版的死锁');
  step(1000);
  assert.equal(rows(),2);
  assert.equal(gateJobs(),1,'第二段之后仍有未显示的段，定时器必须补排');
  assertLocked('第三段出现前');
  step(1000);
  assert.equal(rows(),3);
  assert.equal(gateJobs(),0,'最后一段显示后不再需要定时器');
  release14();
  await p14;
  assertFree('全部段显示完');
  noGate('全部段结束');

  // ── 15. JINE：每段是独立「头像 + 气泡」 ──
  // ⚠️ 顺序要紧：clearMessages() 会因为上一轮还留着 _lastIdentity 补发一条 error 生命周期，
  // 必须在清空 life 之前调用，否则这条 error 会被算进本用例的生命周期断言。
  chat.clearMessages(); clearListeners(); life=[]; setTaskMode('chat');
  const jine=ngo.ChatCore(ctx.jineChannel);
  jine.loadHistory([{user:'U',assistant:'J1\nJ2\nJ3',task_mode:'chat'}],{timeBars:false});
  assert.equal(jineRows().length,3,'JINE 历史三段 = 三条独立行');
  jineRows().forEach(function(r,i){
    assert.equal(r.children.length,2,'JINE 每段是「头像 + 气泡」两项，没有多余空元素');
    assert.equal(r.children[0].className,'jine-bubble-icon','第一项是头像');
    assert.equal(r.children[1].className,'jine-bubble','第二项是气泡');
    assert.equal(r.children[1].textContent,['J1','J2','J3'][i]);
  });
  // JINE 实时流式同样逐段新建「头像 + 气泡」
  clearListeners(); life=[]; jine.clearMessages();
  el('jine-input').value='hi';
  pushEvent(async function(p,m,tm,cb){cb({type:'done',answer:'K1\nK2'});});
  const jp=jine.sendMessage(); jp.catch(function(){});
  await tick();
  assert.equal(jineRows().length,1,'JINE 第一段先出现');
  assertLocked('JINE 第一段');
  step(1000);
  assert.equal(jineRows().length,2,'JINE 第二段到点后出现');
  await jp;
  assert.deepEqual(jineTexts(),['hi','K1','K2']);
  jineRows().forEach(function(r,i){
    assert.equal(r.children.length,2);
    assert.equal(r.children[1].className,'jine-bubble');
    assert.equal(r.children[1].textContent,['K1','K2'][i]);
  });
  assertFree('JINE 分段结束');
  noGate('JINE 分段结束');

  clearListeners(); chat.clearMessages();
  assert.equal(closeWindow!==null,true,'live.js 应已注册直播窗口');
  clearInterval(heartbeat);   // 用例跑完才允许事件循环结束
  console.log('JS segment behavior: 单行零变化 / 换行3段=3条独立消息 / 段间隔>=1s(999与1000对照) / 输入锁到最后一段 / 历史无延时 / work 不分段 / 空段丢弃+\\r\\n / 段内容不trim / 代码围栏保护(闭+未闭合) / 两条卡死回归 / abort 无悬挂定时器 / 字幕逐段切换 / 闸门不变式 / JINE 头像+气泡 全部通过');
})().catch(function(e){clearInterval(heartbeat);console.error(e);process.exitCode=1;});
