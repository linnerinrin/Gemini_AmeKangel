/**
 * services/chat-core — 共享聊天引擎 + 消息 DOM 渲染
 * channel 提供 CSS 类名和图标差异，引擎统一构建 DOM
 */
'use strict';

window.NGO = window.NGO || {};

// ── 表情标签判定 ──
// 数据库里存的是 jine_pien_* 表情标签，重新进入存档时需要还原成表情图片
function safeText(value) {
  return value === null || value === undefined ? '' : String(value);
}

// 中文标签也算表情：模型可能输出 <永远爱你>，历史里也可能留着旧数据
NGO.emojiCodeOf = function(label) {
  if (!label) return null;
  if ((NGO.emojiCodes || []).indexOf(label) !== -1) return label;
  var aliases = NGO.emojiAliases || {};
  return aliases[label] || null;
};

// Keep committed nodes; reparse only the undecided suffix between deltas.
NGO.renderMessageContent = function(target, text, emojiClass, options) {
  var source = safeText(text), final = !options || options.final !== false;
  if (target.style) target.style.whiteSpace = 'pre-wrap';
  var state = target._messageContent;
  if (!state || (state.parts.length && state.parts[0].node.parentNode !== target)) {
    target.textContent = '';
    state = target._messageContent = {source: '', end: 0, parts: [], pending: null};
  }
  if (state.pending) { if (state.pending.parentNode === target) target.removeChild(state.pending); state.pending = null; }
  var common = state.source.length;
  if (source.slice(0, common) !== state.source) {
    common = 0;
    while (common < source.length && common < state.source.length && source[common] === state.source[common]) common++;
    while (state.parts.length && state.parts[state.parts.length - 1].end > common) {
      var old = state.parts[state.parts.length - 1];
      if (old.text && old.start < common) {
        old.node.deleteData(common - old.start, old.end - common); old.end = common; break;
      }
      target.removeChild(old.node); state.parts.pop();
    }
    state.end = state.parts.length ? state.parts[state.parts.length - 1].end : 0;
  }
  function append(start, end, code) {
    if (end <= start) return;
    var prev = state.parts[state.parts.length - 1], node;
    if (!code && prev && prev.text) {
      prev.node.appendData(source.slice(start, end)); prev.end = end;
    } else {
      node = code ? document.createElement('img') : document.createTextNode(source.slice(start, end));
      if (code) { node.className = emojiClass || 'danmaku-emoji'; node.src = NGO.emojiImageSrc(code); node.alt = '表情'; }
      target.appendChild(node); state.parts.push({start: start, end: end, text: !code, node: node});
    }
    state.end = end;
  }
  var pairs = {'<': '>', '【': '】', '[': ']', '(': ')', '（': '）'};
  var codes = NGO.emojiCodes || [], i = state.end, plain = i;
  while (i < source.length) {
    var c = source[i], code = null, end = i + 1;
    if (pairs[c]) {
      var close = i + 1;
      while (close < source.length && '>】])）'.indexOf(source[close]) === -1) close++;
      if (close === source.length && !final) break;
      end = close < source.length ? close + 1 : source.length;
      if (source[close] === pairs[c] && c !== '(' && c !== '（') code = NGO.emojiCodeOf(source.slice(i + 1, close).trim());
    } else if (c === 'j') {
      var tail = source.slice(i), candidate = null;
      for (var k = 0; k < codes.length; k++) {
        if (tail.indexOf(codes[k]) === 0 && (!candidate || codes[k].length > candidate.length)) candidate = codes[k];
      }
      if (!final && codes.some(function(value) { return value.indexOf(tail) === 0; })) break;
      if (candidate && !/[a-z_]/.test(source[i + candidate.length] || '')) { code = candidate; end = i + candidate.length; }
    }
    if (code) { append(plain, i); append(i, end, code); plain = end; }
    i = end;
  }
  append(plain, i);
  if (i < source.length) { state.pending = document.createTextNode(source.slice(i)); target.appendChild(state.pending); }
  state.source = source;
};

NGO.emojiImageSrc = function(code) {
  return BASE + '/assets/emoji/' + code + '.gif';
};

// ── 任务轴分段（聊天模式）──
// 聊天模式按换行分段：收到一个 \n 就等于上一条消息封口（兼容 \r\n）。
// 段内容一律不 trim：行首空格、缩进、段内换行原样保留；
// 整段只有空白时丢弃（连续空行不产生空气泡）。
// 代码围栏保护：某行 trim 后以 ``` 开头即围栏分隔行，从开围栏到配对闭围栏
// 之间的换行都不触发分段，整块（含围栏行本身）算同一个段；
// 流式期间围栏还没等到闭合时按"仍在围栏内"处理，同样先不切。
var SEG_DELAY = 1000;   // 后一段至少等满 1 秒才开始显示

// 一行（不含换行符）是不是围栏分隔行：trim 后以三个反引号开头
function isFenceLine(line) {
  return line.replace(/^[\s]+/, '').indexOf('```') === 0;
}

// 按换行切分（历史回放与流式显示共用），返回全部非空段：
//   - 「已经收到 \n」的那一行算封口，封口的一段就是一条独立消息；
//   - 末尾还没换行的部分是"增长中的末段"，也算一段（流式期间它会继续变长）；
//   - 围栏内的换行不切段；未闭合的围栏整块留在末段里；
//   - 只有空白的段丢弃；段内容原样返回，绝不 trim。
// ⚠️ 段内容一律不 trim：trim() 只用来判断"这一段是不是空的"，
//    入栈的必须是原文（行首空格、缩进、块内换行都得原样保留）。
function splitLineSegments(text) {
  var source = safeText(text), out = [], i = 0, start = 0, len = source.length, inFence = false;
  while (i < len) {
    var j = source.indexOf('\n', i);
    var lineEnd = j === -1 ? len : j;
    var contentEnd = lineEnd;
    if (contentEnd > i && source.charAt(contentEnd - 1) === '\r') contentEnd--;   // \r\n 只吃掉 \r
    if (isFenceLine(source.slice(i, contentEnd))) inFence = !inFence;
    if (j === -1) break;   // 末行还没换行：它是"增长中的末段"，留给循环外用
    if (!inFence) {        // 围栏内不切：整块（含围栏行本身）留在同一个段里
      var piece = source.slice(start, contentEnd);
      if (piece.trim()) out.push(piece);
      start = j + 1;
    }
    i = j + 1;
  }
  var tail = source.slice(start);
  // \r\n 可能只到了一半（流式）：末尾孤立的 \r 不算段内容
  if (tail.length && tail.charAt(tail.length - 1) === '\r') tail = tail.slice(0, -1);
  if (tail.trim()) out.push(tail);
  return out;
}

NGO.ChatCore = function(channel) {
  var _lastBarTime = 0;
  var _viewRevision = 0;
  var _activeRequest = null;
  var _lastIdentity = null;
  var FOLLOW_SLACK = 64;   // 距底部多少像素内算"就在底部"

  function getChatScroller() {
    return document.querySelector(channel.chatAreaSelector);
  }

  function selectingInside(area) {
    var selection = window.getSelection && window.getSelection();
    if (!selection || selection.isCollapsed || !selection.rangeCount) return false;
    var node = selection.anchorNode;
    if (!node) return false;
    var el = node.nodeType === 3 ? node.parentNode : node;
    return !!(el && area.contains && area.contains(el));
  }

  function isNearBottom(area) {
    return area.scrollHeight - area.scrollTop - area.clientHeight <= FOLLOW_SLACK;
  }

  function watchScroller(area) {
    if (!area || area._chatFollowState) return;
    var state = area._chatFollowState = {follow: isNearBottom(area), scheduled: false, scheduledHandle: null, raf: false};
    area.addEventListener('scroll', function() {
      state.follow = isNearBottom(area);
    }, {passive: true});
  }

  function canScroll(area) {
    if (!area) return false;
    watchScroller(area);
    var state = area._chatFollowState;
    if (!selectingInside(area) && isNearBottom(area)) state.follow = true;
    return state.follow && !selectingInside(area);
  }

  // Always follow the configured outer chat viewport; defer until layout has settled.
  function scheduleFollow(area) {
    if (!area || !canScroll(area)) return;
    var state = area._chatFollowState;
    if (state.scheduled) return;
    state.scheduled = true;
    var run = function() {
      state.scheduled = false;
      state.scheduledHandle = null;
      if (!state.follow || selectingInside(area)) return;
      area.scrollTop = Math.max(0, area.scrollHeight - area.clientHeight);
    };
    if (typeof requestAnimationFrame === 'function') {
      state.raf = true;
      state.scheduledHandle = requestAnimationFrame(run);
    } else {
      state.raf = false;
      state.scheduledHandle = setTimeout(run, 0);
    }
  }

  // ── 消息 DOM 渲染 ──
  // 只建行骨架（头像 + 内容容器），内容由 renderMessageContent 填。
  function buildRow(isUser) {
    var msgDiv = document.createElement('div');
    msgDiv.className = channel.msgClass(isUser);

    var iconSrc = isUser ? channel.userIcon() : channel.botIcon();
    if (iconSrc) {
      var icon = document.createElement('img');
      icon.className = channel.iconClass(isUser);
      icon.src = iconSrc;
      msgDiv.appendChild(icon);
    }

    // 内容：有气泡时用气泡当容器，没有气泡（弹幕）时才用 span 直接当内容容器。
    // 两者不能都建——.jine-msg 是 flex+gap，多一个空元素会顶开间距。
    var bubbleClass = channel.bubbleClass(isUser);
    var target;
    if (bubbleClass) {
      target = document.createElement('div');
      target.className = bubbleClass;
    } else {
      target = document.createElement('span');
    }
    msgDiv.appendChild(target);
    msgDiv._contentTarget = target;
    return msgDiv;
  }

  // 是否表情完全由文本决定（renderMessageContent 识别代码与带括号标签），
  // 调用方不再需要额外传 isEmoji 标记。
  // opts.silent = true 时不触发 channel.onRenderMessage（用于回填历史记录）
  function renderMessage(text, isUser, opts) {
    opts = opts || {};
    var area = getChatScroller();
    if (!area) return;

    if (!safeText(text).trim()) return opts.replace || null;

    var scroll = canScroll(area);
    var msgDiv = opts.replace && opts.replace.parentNode === area ? opts.replace : null;
    if (msgDiv && msgDiv._contentTarget) {
      NGO.renderMessageContent(msgDiv._contentTarget, channel.transformText ? channel.transformText(text, isUser) : text, channel.emojiClass(isUser), {final: opts.final !== false});
      if (canScroll(area)) scheduleFollow(area);
      if (channel.onRenderMessage && !opts.silent) channel.onRenderMessage(safeText(text), isUser);
      return msgDiv;
    }
    msgDiv = buildRow(isUser);
    NGO.renderMessageContent(msgDiv._contentTarget, channel.transformText ? channel.transformText(text, isUser) : text, channel.emojiClass(isUser), {final: opts.final !== false});

    if (opts.replace && opts.replace.parentNode === area) area.replaceChild(msgDiv, opts.replace);
    else area.appendChild(msgDiv);
    if (scroll) scheduleFollow(area);
    if (channel.onRenderMessage && !opts.silent) channel.onRenderMessage(safeText(text), isUser);
    return msgDiv;
  }

  // 思考链展开变高时，始终跟随 channel 配置的外层聊天区。
  function followThinking() {
    var area = getChatScroller();
    if (canScroll(area)) scheduleFollow(area);
  }

  // 思考阶段先占一行：思考块挂在这一行上，回答正文稍后填进同一个气泡。
  function createThinkingRow(area) {
    var row = buildRow(false);
    area.appendChild(row);
    if (NGO.thinking) NGO.thinking.begin(row, followThinking);
    return row;
  }

  function cancelFollow(area) {
    if (!area || !area._chatFollowState) return;
    var state = area._chatFollowState;
    if (state.scheduledHandle !== null) {
      if (typeof cancelAnimationFrame === 'function' && state.raf) cancelAnimationFrame(state.scheduledHandle);
      else clearTimeout(state.scheduledHandle);
    }
    state.scheduledHandle = null;
    state.scheduled = false;
  }

  function clearMessages() {
    if (_activeRequest) _activeRequest.cancel();
    if (_lastIdentity && channel.onAnswerLifecycle) channel.onAnswerLifecycle('error', '', _lastIdentity);
    _lastIdentity = null;
    if (NGO.thinking) NGO.thinking.clearAll();
    _viewRevision++;
    var area = document.querySelector(channel.chatAreaSelector);
    if (!area) return;
    area.textContent = '';
  }

  // ── Timebar ──
  function formatHistoryTime(isoString) {
    var d = new Date(isoString);
    if (isNaN(d.getTime())) return '';
    var y = d.getFullYear();
    var mo = String(d.getMonth() + 1).padStart(2, '0');
    var day = String(d.getDate()).padStart(2, '0');
    var h = String(d.getHours()).padStart(2, '0');
    var mi = String(d.getMinutes()).padStart(2, '0');
    return y + '/' + mo + '/' + day + ' ' + h + ':' + mi;
  }

  function insertTimeBar(isoString) {
    var area = document.querySelector(channel.chatAreaSelector);
    if (!area) return;
    var text = formatHistoryTime(isoString);
    if (!text) return;
    var bar = document.createElement('div');
    bar.className = channel.timeBarClass || 'jine-date-bar';
    if (channel.renderTimeBar) {
      channel.renderTimeBar(bar, text);
    } else {
      bar.textContent = text;
    }
    area.appendChild(bar);
  }

  function maybeInsertTimeBar(isoString, isFirst) {
    if (isFirst) { insertTimeBar(isoString); return; }
    if (!isoString) return;
    var ts = new Date(isoString).getTime();
    if (isNaN(ts)) return;
    if (ts - _lastBarTime >= 10 * 60 * 1000) {
      insertTimeBar(isoString);
      _lastBarTime = ts;
    }
  }

  // ── Send ──
  async function doSend(text) {
    if ((channel === window.liveChannel && NGO.liveHistoryLoading) || (channel === window.jineChannel && NGO.jineHistoryLoading)) { if (typeof showError === 'function') showError('正在加载聊天历史，请稍候'); return; }
    if (NGO.chatBusy) { if (typeof showError === 'function') showError('请等待当前回答完成'); return; }
    var token = {}; NGO.chatBusy = token;
    var key = channel.getContextKey ? channel.getContextKey() : null;
    var mode = channel.getMode ? channel.getMode() : '0';
    // 任务轴（chat=闲聊 / work=工作）：P5 才会提供 UI 改 NGO.taskMode，这里先兜底 chat
    var taskMode = channel.getTaskMode ? channel.getTaskMode() : 'chat';
    var generation = NGO.chatSessionGeneration || 0;
    var identity = {generation: generation, key: key, requestId: NGO.chatRequestSequence = (NGO.chatRequestSequence || 0) + 1};
    var controller = new AbortController(); NGO.activeChatController = controller;
    var revision = _viewRevision, answer = '', think = '', node = null, thinkRow = null, trace = [];
    var pendingTimer = null, finished = false, succeeded = false;
    // 分段状态（只在聊天模式、且正文出现换行后才建立）
    // seg.shownAt = 上一段开始显示的时间（Now），闸门 = shownAt + SEG_DELAY
    // seg.timer   = 闸门定时器，与合并重绘的 pendingTimer 严格分开（重绘取消不得动它）
    // seg.shown   = 已显示的段数（由 shownParts 数出来，不手工维护，避免游标错位）
    // seg.hasRow  = 已经为某一"段"建过行（第一段接管思考行，之后每段新建行）
    // ⚠️ 不变式：只要 seg.shown < segParts().length，就必然存在一个已排定的闸门定时器。
    var seg = null, segArm = null, segText = '';
    var segWait = null, segResolve = null;
    function sameSession() { return generation === (NGO.chatSessionGeneration || 0); }
    function visible() { return sameSession() && revision === _viewRevision && NGO.chatBusy === token && (!channel.getContextKey || key === channel.getContextKey()); }
    function lifecycle(type) {
      if (channel.onAnswerLifecycle) channel.onAnswerLifecycle(type, segOn() ? segText : answer, identity);
    }
    function cancelPaint() { if (pendingTimer !== null) { clearTimeout(pendingTimer); pendingTimer = null; } }
    function cancel() {
      // ⚠️ finished 在 done 那一刻就为 true，而分段此时可能还在逐段等待。
      // 所以"释放分段等待者"必须排在 finished 判断之前，否则
      // done 之后 clearMessages()/切主题/换存档/会话代次变化都会让等待者永不 resolve，
      // NGO.chatBusy 永久占用（上一版的第二个卡死）。
      cancelPaint();
      cancelSeg();
      if (finished) return;
      var area = getChatScroller();
      cancelFollow(area);
      lifecycle('error'); finished = true; controller.abort();
    }

    // ── 分段（聊天模式）──
    // 每段都从完整正文里重新切一遍：切分是无副作用的纯函数，不维护任何游标，
    // 因此不存在"预演把游标推过头 → 下一段永远切不出来"这类错位。
    // 显示规则：换行一到，前一段就封口定稿并立刻显示；下一段至少等满 SEG_DELAY 才开始显示，
    // 之后随正文继续增长（已封口的段不再改写）。末尾那段是"增长中的末段"（含未闭合的围栏块）。
    function segOn() { return !!seg; }
    // 懒建立：正文出现换行才进分段模式；工作模式恒不建立。
    // ⚠️ 单行回复（整条没有 \n）必须与改造前完全一致：不建 seg、无闸门、走整条渲染路径。
    function resolveSeg() {
      if (seg) return seg;
      if (taskMode !== 'chat' || answer.indexOf('\n') === -1) return null;
      seg = {shownAt: 0, timer: null, done: false, hasRow: false, shown: 0, finished: false, partsAt: -1, parts: []};
      return seg;
    }
    // 分段期间要显示的全部内容：每段从完整正文重新切一遍，且缓存结果。
    // 按"已经提交到正文的长度"失效，保证流式期间每来一个增量才重算一次。
    // 末尾的"增长中的末段"也在其中（它要原地变长）；收尾后它同样由这里定稿。
    function segParts() {
      if (!seg || seg.partsAt !== answer.length) {
        seg.parts = splitLineSegments(answer);
        seg.partsAt = answer.length;
      }
      return seg.parts;
    }
    // 当前"可以显示"的段（薄封装，语义与 segParts 相同）
    function shownParts() { return segParts(); }
    function cancelSeg() {
      if (segArm !== null) { clearTimeout(segArm); segArm = null; }
      if (seg) seg.timer = null;
      if (segWait) { segWait.reject(Object.assign(new Error('分段等待已取消'), {name: 'AbortError'})); segWait = null; segResolve = null; }
    }
    // 正文完整到达：末段（末尾即使没换行）从此刻起就是定稿的一段
    function finishSeg() {
      if (!seg || seg.done) return;
      seg.done = true;
      seg.partsAt = -1;   // 收尾后要重新算一次（末段可能刚出现/刚变成定稿段）
      armSeg();   // 有未显示的段 → 必须排定定时器
    }
    // 闸门：关着就保证有一个已排定的定时器（不变式）；开着就直接放下一段出来
    function armSeg() {
      if (!seg || seg.shown >= shownParts().length) return;
      if (seg.timer !== null) return;
      var now = Date.now(), wait = seg.shownAt + SEG_DELAY - now;
      if (wait <= 0) { revealSeg(now); return; }
      seg.timer = setTimeout(function() {
        seg.timer = null; segArm = null;
        revealSeg(Date.now());
      }, wait);
      segArm = seg.timer;
    }
    function parkSeg() {
      if (segArm !== null) { clearTimeout(segArm); segArm = null; }
      if (seg) seg.timer = null;
    }
    function revealSeg(now) {
      if (!seg) return;
      if (!visible()) {
        // 已取消/已清空/会话代次变了：释放挂起的等待者，绝不能留着 chatBusy 永久占用
        cancelSeg();
        return;
      }
      var parts = shownParts();
      if (seg.shown >= parts.length) {
        if (seg.done) settleSeg();
        return;
      }
      if (now < seg.shownAt + SEG_DELAY) {   // 闸门还关着 → 必须补排定时器，否则后续段永不出现
        armSeg();
        return;
      }
      parkSeg();   // 闸门重新关闭
      // 第一段接管思考行（思考块挂在它上面），之后每段新建独立「头像 + 气泡」行。
      // 先把当前段画出来，再推进 segText——这样生命周期里的文本正好是这一段。
      var first = !seg.hasRow;
      var text = parts[seg.shown];
      seg.hasRow = true;
      paint(false, {replace: first ? node : null, force: true, text: text});
      segText = text;
      seg.shown++;
      seg.shownAt = Date.now();
      // 每段各发一次 update 让字幕逐段切换；最后一段显示出来才算收尾 done
      if (seg.done && seg.shown >= segParts().length) segDoneNow();
      else lifecycle('update');
      armSeg();
    }
    // done 之后必须等到最后一段真的显示出来；不看定时器，只看"还有没有段没显示"
    function settleSeg() {
      if (!seg || !seg.done || seg.shown < shownParts().length) return;
      parkSeg();
      if (segResolve) { var finish = segResolve; segResolve = null; segWait = null; finish(); }
    }
    function drainSeg() {
      if (!seg || !seg.done) return null;
      // await 的等待对象：cancel()/clearMessages()/会话代次变化/定时器发现 !visible() 都会经 cancelSeg() 释放
      if (!segWait) {
        var rejectWait = null;
        var promise = new Promise(function(res, rej) { segResolve = res; rejectWait = rej; });
        segWait = {promise: promise, reject: rejectWait};
      }
      var pending = segWait.promise;   // 先取出来：settleSeg 成功时会清空 segWait
      settleSeg();
      return pending;
    }
    var request = _activeRequest = {cancel: cancel};
    _lastIdentity = identity;
    function onAbort() {
      cancel();
      if (NGO.thinking && thinkRow) NGO.thinking.drop(thinkRow);
      if (visible() && node && node.parentNode) node.remove();
    }
    controller.signal.addEventListener('abort', onAbort, {once: true});
    function updateTrace(event) {
      if (event.type === 'node' || event.type === 'tool_call') trace.push({kind: event.type, name: event.name});
      else trace.push({kind: 'text', text: event.text || ''});
      if (NGO.thinking && thinkRow) NGO.thinking.update(thinkRow, trace, identity);
    }
    // 分段收尾：必须同时做两件事——
    // 1) 发一次 done 生命周期（字幕收尾）；
    // 2) 释放 done 之后 await 剩余段的等待者。
    // 只发 done 不 settle 就是死锁：最后一段显示后不会再排定时器，revealSeg 不会再来，
    // 等待者永不 resolve → NGO.chatBusy 永久占用（上一版第二个卡死）。
    function segDoneNow() {
      if (!seg || seg.finished) return;
      seg.finished = true;
      parkSeg();           // 没有未显示的段了，闸门定时器必须收掉，不留悬挂定时器
      lifecycle('done');
      settleSeg();         // 输入锁在 doSend 的 finally 里释放
    }
    function paint(final, opts) {
      cancelPaint();
      // 注意：这里不能用 finished 拦截。done 之后还有剩余段要逐段显示，
      // finished 只是"流已收完"，不代表视图不该再画——取消/清空/换会话由 visible() 负责。
      if (!visible()) return;
      var segmented = segOn();
      var force = !!(opts && opts.force);   // revealSeg 正在显示"某一段"
      var text, replace;
      if (segmented && !force) {
        var parts = segParts();
        if (seg.done && seg.shown >= parts.length) {
          // 流已收完且所有段都显示过了（最后一段在流式期间就已经显示并增长过）→ 直接收尾
          segDoneNow();
          if (final && channel.onAnswerLifecycle && channel.onRenderMessage) channel.onRenderMessage(safeText(answer), false, identity);
          return;
        }
        // 普通合并重绘在分段模式下只做一件事：把"已经显示出来、还在继续增长的最后一段"接着画长。
        // 已定稿的段不能再改；还没到点的段也绝不能提前吐出来（那就是闸门失效）。
        var growing = !seg.done && seg.hasRow && parts.length > 0 && seg.shown >= parts.length;
        if (!growing) {
          if (final && channel.onAnswerLifecycle && channel.onRenderMessage) channel.onRenderMessage(safeText(answer), false, identity);
          return;
        }
        text = parts[parts.length - 1];
        segText = text;
        replace = node;   // 原地增长
      } else {
        // 段文本显式传进来（不能读 segText：revealSeg 要在画完之后才推进它）
        text = segmented ? (opts && opts.text != null ? opts.text : segText) : answer;
        replace = opts && opts.hasOwnProperty('replace') ? opts.replace : node;
      }
      node = renderMessage(text, false, {silent: !final || !!channel.onAnswerLifecycle, replace: replace, final: final});
      // 回答正文开始出现 → 思考阶段结束，思考面板自动收起（用户手动展开过则保持）
      if (!final && NGO.thinking && thinkRow) NGO.thinking.collapseForAnswer(thinkRow);
      // 分段时每段的 update/done 由 revealSeg 在段显示后发出（那里才知道段文本）
      if (!segmented) lifecycle(final ? 'done' : 'update');
      else if (!force) lifecycle('update');   // 增长中的那一段：字幕随之更新
      if (final && channel.onAnswerLifecycle && channel.onRenderMessage) channel.onRenderMessage(text, false, identity);
    }
    function schedulePaint() {
      if (pendingTimer === null) pendingTimer = setTimeout(function() { pendingTimer = null; paint(false); }, 16);
    }
    renderMessage(text, true);
    // 先占一行给本轮回复，思考块挂上去，回答正文稍后填进同一个气泡
    var area = document.querySelector(channel.chatAreaSelector);
    if (area) {
      thinkRow = node = createThinkingRow(area);
      if (canScroll(area)) area.scrollTop = area.scrollHeight;
    }
    lifecycle('start');
    channel.clearInput(); channel.playSendSfx();
    if (channel.playThinkingAction) channel.playThinkingAction(key);
    try {
      await NGO.chatAPI.streamChat(text, mode, taskMode, function(event) {
        if (finished) return;
        if (!visible()) { cancel(); return; }
        if (event.type === 'thinking_delta') think += event.text || '';
        if (['node', 'thinking_delta', 'tool_call'].indexOf(event.type) !== -1) updateTrace(event);
        if (event.type === 'answer_reset') {
          // 保留这一行（思考块挂在上面），只清掉上一版草稿正文
          cancelPaint(); cancelSeg(); answer = ''; seg = null; segText = '';
          if (node && node._contentTarget) {
            node._contentTarget.textContent = '';
            node._contentTarget._messageContent = null;
          }
          lifecycle('reset');
        }
        if (event.type === 'answer_delta') {
          answer += event.text || '';
          resolveSeg();
          armSeg();           // 换行一到就立刻把前一段封口定稿显示，并给下一段排定闸门定时器
          schedulePaint();
        }
        if (event.type === 'done') {
          cancelPaint(); answer = event.answer == null ? answer : safeText(event.answer);
          if (!safeText(answer).trim()) throw new Error('模型未返回有效回答，请重试');
          resolveSeg(); finishSeg();
          if (!think) think = event.think || '';
          paint(true); finished = true; succeeded = true;
          if (NGO.thinking && thinkRow) NGO.thinking.finish(thinkRow);
          if (visible()) channel.playReceiveSfx();
        }
        if (event.type === 'error') throw new Error(event.message || '生成失败');
      }, controller.signal);
      if (!finished && visible()) throw new Error('连接中断，回答未完成');
      if (succeeded && visible() && channel.onAfterReply) channel.onAfterReply();
      // 输入锁保持到最后一段出现：这里必须 await 完剩余段才走 finally
      var pendingSeg = drainSeg();
      if (pendingSeg) await pendingSeg;
    } catch (err) {
      cancelPaint();
      if (!finished) lifecycle('error');
      finished = true;
      cancelSeg();
      if (visible()) {
        if (NGO.thinking && thinkRow) NGO.thinking.drop(thinkRow);
        if (node && node.parentNode) node.remove();
        if (err.name !== 'AbortError') {
          renderMessage('发送失败：' + err.message, false, {silent: true});
          if (typeof showError === 'function') showError(err.message || '发送失败，请手动重试');
        }
      }
    } finally {
      cancelPaint();
      cancelSeg();   // 输入锁必须在 finally 释放，不能依赖那个可能被放弃的 Promise
      controller.signal.removeEventListener('abort', onAbort);
      if (NGO.chatBusy === token && channel.stopThinkingAction) channel.stopThinkingAction(key);
      if (_activeRequest === request) _activeRequest = null;
      if (NGO.chatBusy === token) NGO.chatBusy = null;
      if (NGO.activeChatController === controller) NGO.activeChatController = null;
    }
  }

  function sendMessage() {
    var text = channel.getInputValue();
    if (text) return doSend(text);
  }
  function sendEmoji(code) { return doSend(code); }

  // ── History ──
  // opts.timeBars = false 时不插入时间分隔条
  function loadHistory(historyArray, opts) {
    opts = opts || {};
    clearMessages();
    _lastBarTime = 0;
    (historyArray || []).forEach(function(item, i) {
      if (opts.timeBars !== false) maybeInsertTimeBar(item.created_at, i === 0);
      // 每轮历史自带任务轴（chat/work）：聊天模式按换行分段，工作模式整条一个气泡
      var taskMode = item.task_mode === 'work' ? 'work' : 'chat';
      renderMessage(item.user, true, { silent: true });
      // 回放聊天回复时按换行同步拆成多条（每条独立头像+气泡），无延时、不等待；
      // 工作模式或单行回复（没有换行）时整条一个气泡。
      var parts = taskMode === 'chat' ? splitLineSegments(item.assistant) : null;
      if (!parts) parts = [item.assistant];
      var row = null;
      for (var s = 0; s < parts.length; s++) {
        var part = renderMessage(parts[s], false, { silent: true });
        if (!part) continue;
        part._taskMode = taskMode;
        if (!row) row = part;   // 思考链挂在第一段那一行上（默认收起，点标记才展开）
      }
      if (row && NGO.thinking && item.thinking && item.thinking.length) {
        NGO.thinking.restore(row, item.thinking);
      }
      if (item.created_at) {
        var ts = new Date(item.created_at).getTime();
        if (!isNaN(ts)) _lastBarTime = ts;
      }
    });
    var area = getChatScroller();
    if (area && canScroll(area)) scheduleFollow(area);
  }

  // ── Events ──
  function bindEvents() {
    var sendBtn = document.querySelector(channel.sendBtnSelector);
    if (sendBtn) {
      sendBtn.addEventListener('click', function(e) { e.stopPropagation(); sendMessage(); });
    }

    var input = document.querySelector(channel.inputSelector);
    if (input) {
      input.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') { e.stopPropagation(); sendMessage(); }
      });
    }

    if (channel.emojiToggleSelector && channel.emojiPopupSelector) {
      var toggle = document.querySelector(channel.emojiToggleSelector);
      var popup = document.querySelector(channel.emojiPopupSelector);
      if (toggle && popup) {
        toggle.addEventListener('click', function(e) { e.stopPropagation(); popup.classList.toggle('hidden'); });
      }
    }

    if (channel.emojiBtnSelector) {
      document.querySelectorAll(channel.emojiBtnSelector).forEach(function(btn) {
        btn.addEventListener('click', function(e) {
          e.stopPropagation();
          var code = btn.getAttribute('data-emoji');
          if (code) sendEmoji(code);
        });
      });
    }
  }

  return {
    sendMessage: sendMessage,
    sendEmoji: sendEmoji,
    loadHistory: loadHistory,
    clearMessages: clearMessages,
    renderMessage: renderMessage,
    bindEvents: bindEvents
  };
};
