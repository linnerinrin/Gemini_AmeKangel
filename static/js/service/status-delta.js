/**
 * 状态变化浮字：任务管理器与直播面板使用同一套 stats-label / stats-val-area 结构。
 * 浮字挂在 <stats-label> 内部、随文字留在文档流里，因此始终紧贴标签右侧；
 * 每个标签只保留一个 span，2 秒内再次变化时更新文本并从头播放。
 */
'use strict';

(function() {
  function showDelta(label, text, tone) {
    var el = label.querySelector('.stats-delta');
    if (!el) {
      el = document.createElement('span');
      el.className = 'stats-delta';
      el.addEventListener('animationend', function() {
        el.classList.remove('run');
      });
      label.appendChild(el);
    }
    el.textContent = text;
    el.setAttribute('data-tone', tone);
    el.classList.remove('run');
    void el.offsetWidth; // 使同一元素的 CSS 动画可靠地从头播放
    el.classList.add('run');
  }

  NGO.onStatusChange(function(name, oldVal, newVal, opts) {
    if (opts && opts.silent) return;
    if (oldVal == null || newVal == null) return;
    var delta = Number(newVal) - Number(oldVal);
    if (!Number.isFinite(delta) || delta === 0) return;
    var good = name === 'followers' || name === 'affection' ? delta > 0
      : (name === 'stress' || name === 'mental') && delta < 0;
    var magnitude = name === 'followers' ? Math.abs(delta).toLocaleString() : String(Math.abs(delta));
    var text = (delta > 0 ? '+' : '-') + magnitude;
    var cores = document.querySelectorAll('.stats-core[data-stat="' + name + '"]');
    for (var i = 0; i < cores.length; i++) {
      var area = cores[i].closest('.stats-val-area');
      var label = area && area.querySelector('.stats-label');
      if (label) showDelta(label, text, good ? 'good' : 'bad');
    }
  });
})();
