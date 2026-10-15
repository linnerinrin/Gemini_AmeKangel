/**
 * ui/jine — JINE 聊天窗口 (replies + 渲染辅助 + 窗口注册)
 */
'use strict';

var ameReplies = [
  '嗯，是呢', '今天天气真好呀', '得准备直播了', '有点累了', '嘿嘿，谢谢',
  '阿p，加油吧！', '待会一起打游戏吗？', '肚子好饿啊', '今天也要来看直播哦',
  '唔…怎么办呢', '那个不错！', '不行哦，要好好休息', '知道了，我试试',
  '抱歉，没听到', '啊，我知道那个！'
];

var ameEmojis = ['jine_pien_compromise', 'jine_pien_die', 'jine_pien_loveforever', 'jine_pien_nyan', 'jine_pien_ok', 'jine_pien_praise', 'jine_pien_sorry', 'jine_pien_uncare'];

function insertJineDateBar(timeText) {
  var chatArea = document.querySelector('.jine-chat-area');
  if (!chatArea) return;
  var bar = document.createElement('div');
  bar.className = 'jine-date-bar';
  var img = document.createElement('img');
  img.src = BASE + '/assets/images/JINE_date.png';
  img.alt = 'date';
  var span = document.createElement('span');
  span.className = 'jine-date-text';
  span.textContent = timeText || NGO.formatDateTime();
  bar.appendChild(img); bar.appendChild(span);
  chatArea.appendChild(bar);
}

function updateJineDate() {
  var el = document.querySelector('.jine-date-text');
  if (el) el.textContent = NGO.formatDateTime();
}

// ── 实例（jineChannel 在 registry.js 中定义）──
var jineChat = NGO.ChatCore(jineChannel);
NGO.jineChat = jineChat;
jineChat.bindEvents();

// ── Scroll button ──
var jineScrollBtn = document.getElementById('jine-scroll-btn');
if (jineScrollBtn) {
  jineScrollBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    var c = document.querySelector('.jine-chat-area');
    if (c) c.scrollTo({ top: c.scrollHeight, behavior: 'smooth' });
  });
}

// ── Window registration ──
window.NGO = window.NGO || {};
NGO.WindowManager.register(document.getElementById('jine-window'), {
  onFocus: function() {
    if (typeof updateJineDate === 'function') updateJineDate();
  }
});
