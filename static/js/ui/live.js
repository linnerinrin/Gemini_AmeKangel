/**
 * ui/live — 直播弹幕 (channel 配置 + topic 网格 + 窗口注册)
 */
'use strict';

var kangelLiveReplies = [
  '谢谢大家的支持~', '今天也要加油哦', '欢迎新来的观众', '大家点个关注呀', '谢谢礼物！',
  '一起嗨起来~', '阿p最棒了', '今天直播开心吗？', '晚安啦大家', '明天同一时间见'
];
var kangelLiveEmojis = ['jine_pien_compromise', 'jine_pien_loveforever', 'jine_pien_nyan', 'jine_pien_ok', 'jine_pien_praise', 'jine_pien_sorry'];

var streamReplyHideTimer = null;
var streamReplyIdentity = null;
var streamReplyEpoch = 0;

function replyIdentityVisible(identity) {
  return !identity || (identity.generation === (NGO.chatSessionGeneration || 0) &&
    identity.key === liveChannel.getContextKey());
}

// One channel lifecycle, one subtitle owner. Older requests cannot reset newer text.
function streamReplyLifecycle(type, text, identity) {
  if (type === 'start') {
    if (!replyIdentityVisible(identity)) return;
    if (streamReplyIdentity && identity && streamReplyIdentity.requestId > identity.requestId) return;
    clearStreamReply(); streamReplyIdentity = identity; return;
  }
  if (identity !== streamReplyIdentity) return;
  if (type === 'error' || type === 'reset') {
    clearStreamReply();
    if (type === 'reset') streamReplyIdentity = identity;
    return;
  }
  if (!replyIdentityVisible(identity)) { clearStreamReply(); return; }
  if (type === 'update') updateStreamReply(text);
  if (type === 'done') showStreamReply(text, identity);
}

function clearDanmakuDisplay() {
  var list = document.getElementById('live-danmaku');
  if (list) list.textContent = '';
  clearStreamReply();
}

function clearStreamReply() {
  streamReplyIdentity = null; streamReplyEpoch++;
  if (streamReplyHideTimer !== null) {
    clearTimeout(streamReplyHideTimer);
    streamReplyHideTimer = null;
  }
  var overlay = document.getElementById('live-reply-overlay');
  if (overlay) {
    overlay.textContent = ''; overlay._messageContent = null;
    overlay.classList.remove('show');
  }
}

// 流式增量：直接用模型给的增量刷新字幕（不再用逐字定时器，避免和流式打架）。
// 空字符串表示重写/清空，此时隐藏字幕。
function updateStreamReply(text, final) {
  var overlay = document.getElementById('live-reply-overlay');
  if (!overlay) return;
  if (streamReplyHideTimer) { clearTimeout(streamReplyHideTimer); streamReplyHideTimer = null; }
  if (text === null || text === undefined || String(text) === '') {
    overlay.textContent = '';
    overlay.classList.remove('show');
    return;
  }
  NGO.renderMessageContent(overlay, text, 'danmaku-emoji', {final: !!final});
  overlay.classList.add('show');
}

// 回复结束：确定最终文本，并安排自动隐藏
function showStreamReply(text, identity) {
  var overlay = document.getElementById('live-reply-overlay');
  if (!overlay) return;
  updateStreamReply(text, true);
  if (!overlay.classList.contains('show')) return;
  var epoch = ++streamReplyEpoch;
  streamReplyHideTimer = setTimeout(function() {
    if (epoch !== streamReplyEpoch || identity !== streamReplyIdentity) return;
    if (!replyIdentityVisible(identity)) { clearStreamReply(); return; }
    overlay.classList.remove('show'); streamReplyHideTimer = null;
  }, Math.max(5000, String(text).length * 80));
}

// ── 实例（liveChannel 在 registry.js 中定义）──
var liveChat = NGO.ChatCore(liveChannel);
NGO.liveChat = liveChat;
liveChat.bindEvents();

// ── Topic grid ──
var topicConfigs = NGO.topicConfigs || [];

var topicCells = document.querySelectorAll('.topic-cell');
topicCells.forEach(function(cell, i) {
  var cfg = topicConfigs[i];
  if (!cfg) return;
  cell.setAttribute('data-topic', cfg.id);
  cell.style.backgroundImage = 'url(' + BASE + '/assets/images/' + cfg.netachip + ')';
  cell.querySelector('.topic-cell-title').textContent = cfg.title;
  cell.querySelector('.topic-cell-sub').textContent = cfg.sub;
  if (cfg.locked) { cell.classList.add('locked'); } else { cell.classList.add('clickable'); }
});

var hoverTooltip = document.getElementById('hover-tooltip');
var hoverTitle = document.getElementById('hover-tooltip-title');
var hoverSub = document.getElementById('hover-tooltip-sub');

function positionTopicTooltip(el) {
  var rect = el.getBoundingClientRect();
  var cx = rect.left + rect.width / 2;
  var cy = rect.top + rect.height / 2;
  var tw = hoverTooltip.offsetWidth;
  var th = hoverTooltip.offsetHeight;
  hoverTooltip.style.left = (cx - tw - 8) + 'px';
  hoverTooltip.style.top = (cy - th - 8) + 'px';
}

function showTopicTooltip(cell, cfg) {
  if (!hoverTooltip || !hoverTitle || !hoverSub) return;
  hoverTooltip.classList.remove('hidden', 'music-size');
  hoverTitle.textContent = cfg.title;
  hoverSub.textContent = cfg.sub;
  hoverTooltip.style.width = 'clamp(300px, 36vw, 440px)';
  hoverTooltip.style.height = 'clamp(160px, 24vh, 260px)';
  hoverTooltip.classList.add('show');
  positionTopicTooltip(cell);
}

topicCells.forEach(function(cell, i) {
  var cfg = topicConfigs[i];
  if (!cfg || cfg.locked) return;
  cell.addEventListener('mouseenter', function() { showTopicTooltip(cell, cfg); });
  cell.addEventListener('mouseleave', NGO.hideHoverTooltip);
});

document.getElementById('topic-window').addEventListener('click', function(e) {
  var cell = e.target.closest('.topic-cell');
  if (!cell || cell.classList.contains('locked')) return;
  if (NGO.chatBusy) {
    if (typeof showError === 'function') showError('请等待当前回复完成后切换直播');
    return;
  }
  var topicId = parseInt(cell.getAttribute('data-topic'));
  currentTopic = topicConfigs[topicId];
  if (!currentTopic) return;
  var liveWin = document.getElementById('live-window');
  if (!liveWin) return;
  var liveTitle = document.getElementById('live-msg-title');
  if (liveTitle) liveTitle.textContent = currentTopic.title;
  var liveSub = document.getElementById('live-msg-sub');
  if (liveSub) liveSub.textContent = currentTopic.sub;
  liveWin.classList.remove('hidden');
  liveWin.style.display = '';
  NGO.WindowManager.focus(liveWin);
  document.getElementById('topic-window').classList.add('hidden');
  clearDanmakuDisplay();
  // 思考块挂在各自的回复行内，切换直播时随弹幕一起清空
  startLiveAnimation();
  playSfx('Click');

  // 每个直播主题使用独立 mode，避免不同直播（以及 JINE）共用上下文
  var backendMode = currentTopic.mode || NGO.role2idx[currentTopic.charVariant] || '0';
  // 回填该主题之前的上下文（弹幕）
  var loadToken = NGO.liveHistoryToken = (NGO.liveHistoryToken || 0) + 1;
  var sessionGeneration = NGO.chatSessionGeneration || 0;
  NGO.liveHistoryLoading = true;
  NGO.chatAPI.getHistory(backendMode).then(function(history) {
    if (loadToken !== NGO.liveHistoryToken || sessionGeneration !== (NGO.chatSessionGeneration || 0)) return;
    if (!currentTopic || currentTopic.id !== topicId) return;
    NGO.liveChat.loadHistory(history || []);
  }).catch(function(err) {
    if (loadToken === NGO.liveHistoryToken && typeof showError === 'function') showError(err.message || '历史加载失败');
  }).finally(function() {
    if (loadToken === NGO.liveHistoryToken) NGO.liveHistoryLoading = false;
  });
});

// ── Window registration ──
window.NGO = window.NGO || {};
NGO.WindowManager.register(document.getElementById('live-window'), {
  draggable: false,
  onClose: function() {
    clearStreamReply();
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
    if (typeof stopLiveAnimation === 'function') stopLiveAnimation();
  }
});
