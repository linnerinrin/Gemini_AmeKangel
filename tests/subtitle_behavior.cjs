const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const {El,timers}=require('./stream_fake_dom.cjs');
const root=path.resolve(__dirname,'..');
const clock=timers(), ids=new Map();
// 思考点号也用 setTimeout 排队（400ms），所以不能再用"jobs 总数"判断字幕定时器，
// 按延迟区分：<=16ms 是合并重绘，>=5000ms 是字幕自动隐藏。
const pendingPaint=()=>[...clock.jobs.values()].filter(j=>j.delay<=16).length;
const hideTimers=()=>[...clock.jobs.values()].filter(j=>j.delay>=5000).length;
for(const id of ['live-danmaku','live-reply-overlay','topic-window','live-window','live-chat-input'])ids.set(id,new El());
ids.get('live-chat-input').value='hello';let selected={isCollapsed:true,rangeCount:0,anchorNode:null},close;
const ctx=vm.createContext({console,showError:m=>console.error('ERR',m),AbortController,BASE:'/static',currentTopic:{id:0,mode:'1'},liveTimer:null,playSfx(){},requestAnimationFrame:fn=>clock.setTimeout(fn,0),setTimeout:clock.setTimeout,clearTimeout:clock.clearTimeout,clearInterval(){},getSelection:()=>selected,document:{getElementById:id=>ids.get(id)||null,querySelector:s=>s[0]==='#'?ids.get(s.slice(1))||null:null,querySelectorAll:()=>[],createElement:t=>new El(t),createTextNode:t=>{const n=new El('#text');n.textContent=t;return n;}}});
ctx.window=ctx;ctx.NGO={topicConfigs:[],WindowManager:{register(el,opts){close=opts.onClose;}}};
function load(f){vm.runInContext(fs.readFileSync(path.join(root,f),'utf8'),ctx);}
load('static/js/data/registry.js');load('static/js/service/text.js');load('static/js/service/thinking.js');load('static/js/service/chat.js');load('static/js/ui/live.js');
const ngo=ctx.NGO,channel=ctx.liveChannel,chat=ngo.liveChat,overlay=ids.get('live-reply-overlay'),area=ids.get('live-danmaku');
function render(text,final=true,target=overlay){ngo.renderMessageContent(target,text,'emoji',{final});return target;}
(async()=>{
 // Paired delimiters only, including bare-code protection inside malformed/round wrappers.
 for(const t of ['<好]','【永远爱你>','[好】','(好)','（永远爱你）','<jine_pien_ok]','(jine_pien_ok)','<好','永远爱你','jine_pien_ok_extra']){const e=render(t,true,new El());assert.equal(e.textContent,t);assert.equal(e.childNodes.some(n=>n.tag==='img'),false);}
 for(const t of ['<好>','【好】','[好]','jine_pien_ok'])assert.equal(render(t,true,new El()).firstChild.tag,'img');
 assert.equal(ngo.resolveEmojiCode,undefined);
 const e=new El();render('hello',false,e);const txt=e.firstChild;assert.equal(e.style.whiteSpace,'pre-wrap');render('hello world',false,e);assert.equal(e.firstChild,txt);
 render('hello world <永',false,e);assert.equal(e.firstChild,txt);render('hello world <永远爱你>',false,e);assert.equal(e.childNodes[1].tag,'img');const emoji=e.childNodes[1];
 render('hello world <永远爱你> 后续',false,e);render('hello world <永远爱你> 终稿',true,e);assert.equal(e.firstChild,txt);assert.equal(e.childNodes[1],emoji);
 for(const emojiText of ['<永远爱你>','【好】','[jine_pien_ok]','jine_pien_ok'])for(let split=1;split<emojiText.length;split++){const target=new El();render('prefix'+emojiText.slice(0,split),false,target);render('prefix'+emojiText,false,target);render('prefix'+emojiText,true,target);assert.equal(target.textContent,'prefix');assert.equal(target.childNodes.filter(n=>n.tag==='img').length,1);}
 const history=[{user:'hello',assistant:'history'}];chat.loadHistory(history,{timeBars:false});assert.equal(overlay.classList.contains('show'),false);
 let events=[],traceCalls=[];const taskModes=[];const life=channel.onAnswerLifecycle,updateThink=ngo.thinking.update;
 channel.onAnswerLifecycle=function(type,t,id){events.push(type);return life.call(this,type,t,id);};
 // 思考链现在直接内联到行上，这里观察生产路径（同一数组只追加、identity 稳定）
 ngo.thinking.update=function(row,trace,id){traceCalls.push({t:trace,id:id,length:trace.length});return updateThink.call(this,row,trace,id);};
 ngo.chatAPI={streamChat:async(p,m,tm,cb)=>{taskModes.push(tm);cb({type:'thinking_delta',text:'first'});cb({type:'node',name:'answer'});cb({type:'answer_delta',text:'hel'});cb({type:'answer_delta',text:'lo'});assert.equal(overlay.classList.contains('show'),false);clock.flush(16);assert.equal(overlay.textContent,'hello');const message=area.lastChild,node=message._contentTarget.firstChild,subnode=overlay.firstChild;cb({type:'answer_delta',text:' world'});clock.flush(16);assert.equal(area.lastChild,message);assert.equal(message._contentTarget.firstChild,node);assert.equal(overlay.firstChild,subnode);cb({type:'answer_delta',text:' discarded'});cb({type:'answer_reset'});assert.equal(overlay.textContent,'');clock.flush(0);assert.equal(pendingPaint(),0);cb({type:'answer_delta',text:'new'});cb({type:'done',answer:'new final'});assert.equal(overlay.textContent,'new final');}};
 await chat.sendEmoji('jine_pien_ok');assert.deepEqual(events,['start','update','update','reset','done']);assert.equal(hideTimers(),1);
 // 任务轴（chat/work）要跟 mode 一起到 api 层；liveChannel 没写 NGO.taskMode 时兜底 chat
 assert.deepEqual(taskModes,['chat'],'缺省任务轴要兜底成 chat');
 assert.equal(traceCalls[0].t,traceCalls[1].t);assert.equal(traceCalls[0].id,traceCalls.at(-1).id);assert.equal(traceCalls.at(-1).length,2);assert.equal(traceCalls[0].length,1);
 const oldIdentity=traceCalls[0].id,oldHide=[...clock.jobs.values()].find(j=>j.delay>=5000).fn;
 events=[];ngo.chatAPI.streamChat=async(p,m,tm,cb)=>{cb({type:'answer_delta',text:'draft'});clock.flush(16);throw Error('offline');};await chat.sendEmoji('x');clock.flush(0);assert.equal(overlay.textContent,'');assert.equal(pendingPaint(),0);assert.equal(hideTimers(),0);assert.deepEqual(events,['start','update','error']);
 // Old request/reset/error/timer cannot mutate the new request.
 const fresh={generation:0,key:0,requestId:999};life('start','',fresh);life('update','fresh',fresh);oldHide();life('reset','',oldIdentity);life('error','',oldIdentity);life('done','old',oldIdentity);assert.equal(overlay.textContent,'fresh');
 ctx.currentTopic={id:1};life('update','stale',fresh);assert.equal(overlay.textContent,'');ctx.currentTopic={id:0};
 const next={generation:0,key:0,requestId:1000};life('start','',next);life('done','next',next);ctx.NGO.chatSessionGeneration=1;clock.flush();assert.equal(overlay.textContent,'');
 // Cancellation clears visible draft and its scheduled paint without failure subtitle.
 let cb,release;ngo.chatAPI.streamChat=async(p,m,tm,fn)=>{cb=fn;await new Promise(r=>release=r);};const pending=chat.sendEmoji('x');cb({type:'answer_delta',text:'cancel me'});clock.flush(16);cb({type:'answer_delta',text:'later'});chat.clearMessages();assert.equal(overlay.textContent,'');assert.equal(clock.jobs.size,0);cb({type:'done',answer:'stale'});release();await pending;assert.equal(area.childNodes.length,0);
 // External AbortController cleanup is immediate even before transport settles.
 const aborting=chat.sendEmoji('abort');cb({type:'answer_delta',text:'visible draft'});clock.flush(16);cb({type:'answer_delta',text:'queued'});ngo.activeChatController.abort();assert.equal(overlay.textContent,'');assert.equal(clock.jobs.size,0);release();await aborting;
 // Real old/new requests overlap after a session switch; old completion cannot clear the new subtitle.
 const oldPending=chat.sendEmoji('old');const oldCallback=cb,oldRelease=release;oldCallback({type:'answer_delta',text:'old draft'});clock.flush(16);ngo.chatSessionGeneration++;ngo.chatBusy=null;
 const newPending=chat.sendEmoji('new');const newCallback=cb,newRelease=release;newCallback({type:'answer_delta',text:'new draft'});clock.flush(16);oldCallback({type:'answer_reset'});oldCallback({type:'done',answer:'old done'});oldRelease();await oldPending;assert.equal(overlay.textContent,'new draft');newCallback({type:'done',answer:'new final'});newRelease();await newPending;assert.equal(overlay.textContent,'new final');
 chat.loadHistory(history,{timeBars:false});clock.flush(0);assert.equal(overlay.textContent,'');assert.equal(clock.jobs.size,0);
 // Topic changes reject subsequent transport callbacks and clear the old owned subtitle.
 const topicPending=chat.sendEmoji('topic');cb({type:'answer_delta',text:'topic draft'});clock.flush(16);ctx.currentTopic={id:1};cb({type:'answer_delta',text:'must not paint'});assert.equal(overlay.textContent,'');assert.equal(clock.jobs.size,0);release();await topicPending;ctx.currentTopic={id:0};
 for(const terminal of [{type:'error',message:'protocol failure'},{type:'done',answer:'   '}]){ngo.chatAPI.streamChat=async(p,m,tm,fn)=>{fn({type:'answer_delta',text:'queued draft'});fn(terminal);};await chat.sendEmoji('error');clock.flush(0);assert.equal(overlay.textContent,'');assert.equal(clock.jobs.size,0);}
 // 思考块内联到当轮回复行（走真实 liveChannel + chat.js 生产链路）
 const colOf=row=>row.childNodes.find(n=>n.className&&n.className.includes('msg-col'));
 const blockOf=row=>{const col=colOf(row);return col?col.childNodes.find(n=>n.className&&n.className.includes('think-block')):null;};
 const panelOf=row=>{const block=blockOf(row);return block?block.childNodes.find(n=>n.className&&n.className.includes('think-panel')):null;};
 let thinkRow=null;
 // 窗口跟随：思考链变长时外层容器要滚到底，滚上去看历史时不能被硬拉回来。
 // 断言一律放在 mock 外面：mock 里断言失败会被 doSend 的 catch 吞掉，只剩"答案没写进气泡"这种假象。
 const snap=[];
 const at=name=>snap.find(s=>s.at===name);
 area.scrollHeight=2000;area.clientHeight=100;area.scrollTop=1900;
 ngo.chatAPI.streamChat=async(p,m,tm,cb2)=>{
   cb2({type:'node',name:'agent'});
   cb2({type:'thinking_delta',text:'先想一下'});
   thinkRow=area.lastChild;
   clock.flush(0);
  snap.push({at:'thinking',scroll:area.scrollTop,scrollHeight:area.scrollHeight,
     dots:thinkRow._contentTarget.firstChild&&thinkRow._contentTarget.firstChild.className,
     state:blockOf(thinkRow).getAttribute('data-state'),
     panelHidden:panelOf(thinkRow).classList.contains('hidden'),
     panelText:panelOf(thinkRow).textContent});
   clock.flush(400);
   snap.push({at:'dots',text:thinkRow._contentTarget.firstChild.textContent});
   area.scrollTop=100;
   cb2({type:'thinking_delta',text:'继续想'});
   snap.push({at:'userScrolled',scroll:area.scrollTop});
   area.scrollTop=area.scrollHeight;
   cb2({type:'answer_delta',text:'答案'});
   clock.flush(16);
   snap.push({at:'answer',sameRow:area.lastChild===thinkRow,paints:pendingPaint(),
     panelHidden:panelOf(thinkRow).classList.contains('hidden')});
   cb2({type:'done',answer:'答案'});
   snap.push({at:'done',state:blockOf(thinkRow).getAttribute('data-state'),
     dots:thinkRow._thinkDots,target:thinkRow._contentTarget.textContent});
   ngo.thinking.toggle(thinkRow);
   snap.push({at:'expand',panelHidden:panelOf(thinkRow).classList.contains('hidden'),
     text:panelOf(thinkRow).textContent});
   ngo.thinking.toggle(thinkRow);
   snap.push({at:'collapse',panelHidden:panelOf(thinkRow).classList.contains('hidden')});
 };
 await chat.sendEmoji('inline think');
 assert.equal(at('thinking').scroll,Math.max(0,at('thinking').scrollHeight-area.clientHeight),'思考增量要把窗口带到最底端');
 assert.equal(at('thinking').dots,'think-pending');
 assert.equal(at('thinking').state,'thinking');
 assert.equal(at('thinking').panelHidden,false,'思考中默认展开，要能看见思考过程');
 assert.match(at('thinking').panelText,/节点: agent/);
 assert.match(at('thinking').panelText,/先想一下/);
 assert.equal(at('dots').text,'..','点号要按 1→3 走');
 assert.equal(at('userScrolled').scroll,100,'用户翻上去看历史时不要打扰');
 assert.equal(at('answer').sameRow,true,'回答要填进同一行，不能另起一行');
 assert.equal(at('answer').panelHidden,true,'回答开始后思考收起');
 assert.equal(at('answer').paints,1,'答案渲染后应有一个合并的外层滚动任务');
 assert.equal(at('done').state,'done');
 assert.equal(at('done').dots,null,'结束后点号要停');
 assert.equal(at('done').target,'答案');
 assert.equal(at('expand').panelHidden,false);
 assert.match(at('expand').text,/节点: agent/);
 assert.equal(at('collapse').panelHidden,true);
 assert.equal(ids.get('think-text-area'),undefined,'思考窗口应已移除');
 assert.equal(ids.get('live-important-chat'),undefined,'直播思考面板应已移除');
 // 历史回填：带 thinking 的历史行要长成同样的折叠块（默认收起，点开才有内容）
 // 注意 loadHistory 会先清空消息区，所以历史行从索引 0 开始
 chat.loadHistory([
   {user:'早',assistant:'早安',thinking:[{kind:'node',name:'agent'},{kind:'text',text:'历史思考'}]},
   {user:'嗯',assistant:'好的'},
 ],{timeBars:false});
 const withThink=area.childNodes[1];                // 第一条的机器人行
 const bareRow=area.childNodes[3];                  // 第二条没有思考
 assert.equal(blockOf(withThink).getAttribute('data-state'),'done');
 assert.equal(panelOf(withThink).classList.contains('hidden'),true,'历史思考默认收起');
 assert.equal(panelOf(withThink).textContent,'','收起时不该白渲染');
 assert.equal(blockOf(bareRow),null,'没有思考的历史行不该多出空壳');
 ngo.thinking.toggle(withThink);
 assert.match(panelOf(withThink).textContent,/历史思考/);
 assert.match(panelOf(withThink).textContent,/节点: agent/);
 ngo.thinking.toggle(withThink);
 assert.equal(panelOf(withThink).classList.contains('hidden'),true);
 // Scroll only near the bottom and without a text selection inside this area.
 area.scrollHeight=1000;area.clientHeight=100;area.scrollTop=0;chat.renderMessage('no jump',true);assert.equal(area.scrollTop,0);
 area.scrollTop=900;selected={isCollapsed:false,rangeCount:1,anchorNode:area.firstChild};chat.renderMessage('selected',true);assert.equal(area.scrollTop,900);
 // 别处的残留选区（比如输入框、别的窗口）不该把跟随永久关掉
 selected={isCollapsed:false,rangeCount:1,anchorNode:{}};chat.renderMessage('still follows',true);clock.flush(0);assert.equal(area.scrollTop,900);
 selected={isCollapsed:true,rangeCount:0,anchorNode:null};chat.renderMessage('follow',true);clock.flush(0);assert.equal(area.scrollTop,900);
 // 一次涨很多也要跟得住（旧实现用 64px 阈值判断，跳一大段就会掉队）
 area.scrollHeight=5000;chat.renderMessage('big jump',true);clock.flush(0);assert.equal(area.scrollTop,4900);
 close();assert.equal(clock.jobs.size,0);console.log('subtitle behavior: real live/registry lifecycles, batching, text-node reuse, emoji splits, identity isolation, timer cancellation and scroll passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
