const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const {El,timers}=require('./stream_fake_dom.cjs');
function load(ctx,file){vm.runInContext(fs.readFileSync(path.join(root,file),'utf8'),ctx);}
(async()=>{
 const clock=timers();
 const area=new El(), important=new El(), jineArea=new El();
 const ctx=vm.createContext({console,setTimeout:clock.setTimeout,clearTimeout:clock.clearTimeout,TextDecoder,AbortController,Uint8Array,Response,ReadableStream,BASE:'/static',currentTopic:{id:0},document:{querySelector:s=>s==='#chat'?area:(s==='#jine'?jineArea:null),getElementById:()=>null,createElement:t=>new El(t),createTextNode:t=>{const el=new El('#text');el.textContent=t;return el;}},showError:()=>{}});
 ctx.window=ctx;ctx.location={port:'5577',origin:'http://localhost:5577'};ctx.NGO={emojiCodes:['jine_pien_ok','jine_pien_loveforever'],emojiAliases:{'永远爱你':'jine_pien_loveforever','好':'jine_pien_ok'}};
 load(ctx,'static/js/data/api/chat.js');
 const wire=': ping\r\n\r\ndata: '+JSON.stringify({type:'thinking_delta',text:'你好'})+'\r\n\r\ndata: '+JSON.stringify({type:'done',answer:'完成'})+'\n\n';
 const bytes=new TextEncoder().encode(wire);
 // 记下真正发出去的请求体：任务轴必须跟着 mode 一起送到后端
 const bodies=[];
 ctx.fetch=async(url,opts)=>{bodies.push(JSON.parse(opts.body));return new Response(new ReadableStream({start(c){for(const byte of bytes)c.enqueue(Uint8Array.of(byte));c.close();}}),{headers:{'content-type':'text/event-stream'}});};
 const events=[];await ctx.NGO.chatAPI.streamChat('x','2','work',e=>events.push(e));assert.equal(events.length,2);assert.equal(events[0].text,'你好');
 assert.deepEqual(bodies[0],{prompt:'x',mode:'2',task_mode:'work'});
 ctx.fetch=async(url,opts)=>{bodies.push(JSON.parse(opts.body));return new Response('data: {"type":"answer_delta","text":"x"}\n\n',{headers:{'content-type':'text/event-stream'}});};
 await assert.rejects(ctx.NGO.chatAPI.streamChat('x','2',null,()=>{}),/连接中断/);
 assert.equal(bodies[1].task_mode,null); // null 交给后端沿用会话当前任务轴
 load(ctx,'static/js/service/text.js');
 load(ctx,'static/js/service/chat.js');
 let key=0, stops=0, callback, requests=0;
 const taskModes=[];
 const channel={chatAreaSelector:'#chat',msgClass:()=>'',userIcon:()=>'',botIcon:()=>'',bubbleClass:()=>'',emojiClass:()=>'',getContextKey:()=>key,getMode:()=>String(key+1),getInputValue:()=> 'hello',clearInput(){},playSendSfx(){},playReceiveSfx(){},playThinkingAction(){},stopThinkingAction(){stops++;}};
 const chat=ctx.NGO.ChatCore(channel);
 chat.loadHistory([{user:'<jine_pien_ok>',assistant:'【jine_pien_ok】'}],{timeBars:false});assert.equal(area.children[0].children[0].children[0].src,'/static/assets/emoji/jine_pien_ok.gif');assert.equal(area.children[1].children[0].children[0].src,'/static/assets/emoji/jine_pien_ok.gif');
 ctx.NGO.chatAPI.streamChat=async(p,m,tm,cb)=>{taskModes.push(tm);requests++;cb({type:'answer_delta',text:'bad'});cb({type:'answer_reset'});cb({type:'answer_delta',text:'new'});cb({type:'done',answer:'【jine_pien_ok】'});};
 await chat.sendMessage();assert.equal(area.children.at(-1).children[0].children[0].src,'/static/assets/emoji/jine_pien_ok.gif');assert.equal(stops,1);
 ctx.NGO.chatAPI.streamChat=async(p,m,tm,cb)=>{taskModes.push(tm);requests++;cb({type:'done',answer:'no delta'});};await chat.sendMessage();assert.equal(area.children.at(-1).textContent,'no delta');
 ctx.NGO.chatAPI.singleChat=()=>{throw Error('must not replay');};ctx.NGO.chatAPI.streamChat=async(p,m,tm)=>{taskModes.push(tm);requests++;throw Error('offline');};await chat.sendMessage();assert.match(area.children.at(-1).textContent,/offline/);assert.equal(requests,3);
 // 任务轴（chat/work）必须跟 mode 一样传下去；channel 没提供 getTaskMode 时兜底 chat
 assert.equal(taskModes.every(t=>t==='chat'),true,'缺省任务轴要兜底成 chat');
 channel.getTaskMode=()=>'work';
 ctx.NGO.chatAPI.streamChat=async(p,m,tm,cb)=>{taskModes.push(tm);cb({type:'done',answer:'work reply'});};
 await chat.sendMessage();assert.equal(taskModes.at(-1),'work');assert.equal(area.children.at(-1).textContent,'work reply');
 delete channel.getTaskMode;
 let release;ctx.NGO.chatAPI.streamChat=async(p,m,tm,cb)=>{taskModes.push(tm);callback=cb;await new Promise(r=>release=r);};const pending=chat.sendMessage();await chat.sendMessage();key=1;chat.clearMessages();callback({type:'answer_delta',text:'wrong topic'});callback({type:'done',answer:'wrong'});release();await pending;assert.equal(area.children.length,0);
 key=0;const pending2=chat.sendMessage();ctx.NGO.chatSessionGeneration=1;chat.clearMessages();callback({type:'done',answer:'old save'});release();await pending2;assert.equal(area.children.length,0);
 for (const mode of ['0','1','2','3']) {
   channel.getMode=()=>mode; channel.botIcon=()=>'/avatar.png'; channel.iconClass=()=> 'avatar';
   // 聊天模式已改为按换行分段（全覆盖见 segment_behavior.cjs）：多行回复会排定 1 秒闸门，
   // 而本套件的假定时器从不推进，await 会永久挂住。这里只验模式透传与「头像+正文」，故用单行。
   const text='你说好，我就懒得搭理了。';
   ctx.NGO.chatAPI.streamChat=async(p,m,tm,cb)=>{taskModes.push(tm);assert.equal(m,mode);cb({type:'node',name:'reflection'});cb({type:'done',answer:text});};
    await chat.sendMessage(); assert.equal(area.children.at(-1).textContent,text); assert.equal(area.children.at(-1).children[0].src,"/avatar.png");
    chat.loadHistory([{user:text,assistant:text}],{timeBars:false}); assert.equal(area.children[0].textContent,text); assert.equal(area.children[1].textContent,text);
 }
 // 多行正文的整条渲染：工作模式不按换行分段，换行原样留在同一条消息里
 channel.getTaskMode=()=>'work';
 const multiline='你说好，我就懒得搭理了。\n有意见？\n你"[\\]';
 ctx.NGO.chatAPI.streamChat=async(p,m,tm,cb)=>{assert.equal(tm,'work');cb({type:'done',answer:multiline});};
 await chat.sendMessage(); assert.equal(area.children.at(-1).textContent,multiline);
 chat.loadHistory([{user:multiline,assistant:multiline,task_mode:'work'}],{timeBars:false});
 assert.equal(area.children[0].textContent,multiline); assert.equal(area.children[1].textContent,multiline);
 delete channel.getTaskMode;
 const overlay=new El();ctx.NGO.renderMessageContent(overlay,'好呀<jine_pien_ok>再见','emoji');assert.equal(overlay.textContent,'好呀再见');assert.equal(overlay.children[1].src,'/static/assets/emoji/jine_pien_ok.gif');
 // 中文标签（模型/历史旧数据）也要渲染成表情图片
 const label=new El();ctx.NGO.renderMessageContent(label,'<永远爱你>','emoji');assert.equal(label.children[0].src,'/static/assets/emoji/jine_pien_loveforever.gif');
 const mixed=new El();ctx.NGO.renderMessageContent(mixed,'好呀【永远爱你】','emoji');assert.equal(mixed.textContent,'好呀');assert.equal(mixed.children[1].src,'/static/assets/emoji/jine_pien_loveforever.gif');
 // 裸中文标签不算表情：否则角色正常说一句"抱歉"会被当成表情，直播字幕会被整条吞掉
 // 删掉 isEmoji 参数后：表情靠文本识别仍要渲染成图片，且失败提示必须保持 silent
 // （如果参数位置错位，'发送失败' 那条会把 opts 顶掉，从而多触发一次 onRenderMessage）
 let rendered=0;
 const flagChannel=Object.assign({},channel,{onRenderMessage:()=>{rendered++;}});
 const flagChat=ctx.NGO.ChatCore(flagChannel);
 ctx.NGO.chatAPI.streamChat=async()=>{throw Error('boom');};
 flagChat.clearMessages();
 await flagChat.sendEmoji('jine_pien_ok');
 // 表情消息：div > span > img（有气泡的 channel 则是 div > .bubble > img）
 assert.equal(area.children[0].children[0].children[0].src,'/static/assets/emoji/jine_pien_ok.gif');
 assert.match(area.children[1].textContent,/发送失败/); // 流式接口抛错 → 紧跟一条失败提示
 await flagChat.sendMessage();
 assert.match(area.children.at(-1).textContent,/发送失败/);
 assert.equal(rendered,2); // 用户表情 + 用户文字各 1 次；两条失败提示都是 silent，不计入
 assert.equal(ctx.NGO.isEmojiCode,undefined); // 死代码已移除
 // JINE 形状（有气泡容器）：不能多出空 flex 项，否则 .jine-msg 的 gap 会顶开间距
 const jineChannel=Object.assign({},channel,{chatAreaSelector:'#jine',bubbleClass:()=> 'jine-bubble',userIcon:()=> '/bear.png',botIcon:()=> '/ame.png',iconClass:()=> 'icon',emojiClass:()=> 'jine-emoji-img'});
 const jineChat=ctx.NGO.ChatCore(jineChannel);
 jineChat.loadHistory([{user:'抱歉',assistant:'<永远爱你>'}],{timeBars:false});
 assert.equal(jineArea.children[0].children.length,2); // 头像 + 气泡，没有多余空 span
 assert.equal(jineArea.children[0].children[1].className,'jine-bubble');
 assert.equal(jineArea.children[0].textContent,'抱歉'); // 裸中文标签保持文字
 assert.equal(jineArea.children[1].children[1].children[0].src,'/static/assets/emoji/jine_pien_loveforever.gif');
 ctx.NGO.chatAPI.streamChat=async(p,m,tm,cb)=>{taskModes.push(tm);cb({type:'done',answer:'<永远爱你>'});};
 await jineChat.sendMessage();
 assert.equal(jineArea.children.at(-1).children.length,2);
 assert.equal(jineArea.children.at(-1).children[1].children[0].src,'/static/assets/emoji/jine_pien_loveforever.gif');
 // 表情不再靠"跳过字幕"来处理：字幕用 renderMessageContent 渲染，表情会显示成图片
 // 直播字幕统一由生命周期接口通知，短间隔增量可以合并；
 // 被反思打回（answer_reset）时清空字幕；只有最终渲染才走 onRenderMessage
 // （onRenderMessage 只处理角色消息，用户自己发的消息不算字幕——与 registry 一致）
 const seen=[];
 const subChannel=Object.assign({},channel,{
   onAnswerLifecycle:(type,t)=>{if(type==='reset')seen.push('');if(type==='update')seen.push(t);},
   onRenderMessage:(t,isUser)=>{ if(!isUser) seen.push('FINAL:'+t); }
 });
 const subChat=ctx.NGO.ChatCore(subChannel);
 ctx.NGO.chatAPI.streamChat=async(p,m,tm,cb)=>{
   taskModes.push(tm);
   cb({type:'answer_delta',text:'你'});
   cb({type:'answer_delta',text:'好'});
   cb({type:'answer_reset'});
   cb({type:'answer_delta',text:'新'});
   cb({type:'done',answer:'新答案'});
 };
 subChat.clearMessages();
 await subChat.sendMessage();
 assert.deepEqual(seen,['','FINAL:新答案']); // reset/done cancel queued intermediate paints
 // 思考链：节点/工具加粗独占一行，文本归属到该节点分组内
 ctx.NGO.renderThinking(important,[{kind:'node',name:'reflection'},{kind:'tool_call',name:'search_web'},{kind:'text',text:'简短中文'}]);
 assert.equal(important.children[0].children[0].tag,'strong');
 assert.equal(important.children[0].children[0].textContent,'节点: reflection');
 assert.equal(important.children[2].children[0].tag,'strong');
 assert.equal(important.children[2].children[0].textContent,'工具: search_web');
 assert.equal(important.children[3].textContent,'简短中文');
 // 思考链不再被截断、也不再剥离英文（debug 13）
 const long=[];for(let i=0;i<40;i++)long.push({kind:'text',text:'abc思考内容'+i});
 ctx.NGO.renderThinking(important,long);assert.equal(important.textContent.includes('思考内容39'),true);assert.equal(important.textContent.includes('abc'),true);
 // 历史回填要读出每轮的 task_mode：本阶段只挂到行上备用（P4 才按它分段渲染），默认兜底 chat
 chat.loadHistory([{user:'q',assistant:'a',task_mode:'work'},{user:'q2',assistant:'a2'}],{timeBars:false});
 assert.equal(area.children[1]._taskMode,'work');
 assert.equal(area.children[3]._taskMode,'chat');
 console.log('JS stream behavior: four modes, task mode plumbing, multiline history, avatar+text, emoji labels, node grouping, no truncation, framing and isolation passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
