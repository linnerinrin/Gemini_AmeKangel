/*
 * service/thinking — 内联思考块
 *
 * 思考链不再单独开窗口，而是挂在当轮回复那一行里：
 *   思考中：气泡内显示 1~3 个点的省略号动画，思考内容默认折叠
 *   结束后：气泡下方留一条淡色标记，单击标记展开/收回思考链
 *
 * 注意：气泡文字要保持"双击选中单词"的能力，所以这里只在标记条上绑单击，
 * 不在气泡上绑任何 dblclick。
 */
(function() {
  'use strict';

  var NGO = window.NGO = window.NGO || {};

  var DOTS = ['.', '..', '...'];
  var TICK = 400;
  var dotsTimer = null;
  var dotsStep = 0;
  var dotsEntries = [];

  // 用 setTimeout 排队而不是 setInterval：测试里的假时钟只实现了 setTimeout/clearTimeout。
  function scheduleTick() {
    if (dotsTimer !== null) return;
    dotsTimer = setTimeout(function() {
      dotsTimer = null;
      tick();
      if (dotsEntries.length) scheduleTick();
    }, TICK);
  }

  function tick() {
    dotsStep = (dotsStep + 1) % DOTS.length;
    // 回答正文开始流式后会清空气泡内容，点号节点随之脱离文档，这里顺手摘掉。
    dotsEntries = dotsEntries.filter(function(entry) {
      if (!entry.el || !entry.el.parentNode) return false;
      entry.el.textContent = DOTS[dotsStep];
      return true;
    });
  }

  function startDots(row, el) {
    dotsStep = 0;
    el.textContent = DOTS[dotsStep];
    dotsEntries.push({row: row, el: el});
    scheduleTick();
  }

  function removeDots(row) {
    dotsEntries = dotsEntries.filter(function(entry) {
      if (entry.row !== row) return true;
      if (entry.el && entry.el.parentNode) entry.el.parentNode.removeChild(entry.el);
      return false;
    });
    // 没有任何可见点号时就别再排队了，避免留下空转的定时器
    if (!dotsEntries.length) stopDots();
  }

  function stopDots() {
    if (dotsTimer !== null) { clearTimeout(dotsTimer); dotsTimer = null; }
  }

  function ensureBlock(row) {
    if (row._thinkBlock) return row._thinkBlock;
    var target = row._contentTarget;
    if (!target || !target.parentNode) return null;
    // 思考块要显示在气泡下方，而 .jine-msg/.danmaku-item 都是横向 flex，
    // 所以这里把「气泡 + 思考块」包成一个纵向容器，气泡位置不变。
    var col = document.createElement('div');
    col.className = 'msg-col';
    target.parentNode.replaceChild(col, target);
    col.appendChild(target);

    var block = document.createElement('div');
    block.className = 'think-block';
    block.setAttribute('data-state', 'thinking');

    var toggle = document.createElement('div');
    toggle.className = 'think-toggle';
    toggle.setAttribute('role', 'button');
    toggle.setAttribute('tabindex', '0');
    toggle.setAttribute('aria-expanded', 'false');
    toggle.textContent = '思考过程';

    var panel = document.createElement('div');
    panel.className = 'think-panel hidden';

    function activate(e) { e.stopPropagation(); togglePanel(row); }
    toggle.addEventListener('click', activate);
    toggle.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') { e.preventDefault(); activate(e); }
    });

    block.appendChild(toggle);
    block.appendChild(panel);
    col.appendChild(block);

    row._thinkCol = col;
    row._thinkBlock = block;
    row._thinkToggle = toggle;
    row._thinkPanel = panel;
    row._thinkState = {trace: [], identity: null};
    return block;
  }

  function renderPanel(row) {
    if (!row._thinkPanel) return;
    NGO.renderThinking(row._thinkPanel, row._thinkState.trace, row._thinkState.identity);
  }

  // 思考面板变高以后，外层聊天窗口要跟着往下走，否则新内容全在可视区外。
  // 具体能不能滚由调用方决定（用户往上翻或正在选字时不该硬拉回去）。
  function follow(row) {
    if (row._thinkScroll) row._thinkScroll(row);
  }

  function setExpanded(row, expand) {
    var panel = row._thinkPanel, toggle = row._thinkToggle;
    if (!panel || !toggle) return;
    if (expand) { renderPanel(row); panel.classList.remove('hidden'); }
    else { panel.classList.add('hidden'); }
    toggle.setAttribute('aria-expanded', expand ? 'true' : 'false');
    if (row._thinkBlock) row._thinkBlock.setAttribute('data-expanded', expand ? 'true' : 'false');
    if (expand) follow(row);
  }

  function togglePanel(row) {
    var panel = row._thinkPanel;
    if (!panel) return false;
    var expand = panel.classList.contains('hidden');
    row._thinkUserToggled = true;   // 用户自己点过，之后不再自动展开/收起
    setExpanded(row, expand);
    return expand;
  }

  // 没有思考内容时把气泡收回原位，避免每行都多一层空容器
  function unwrap(row) {
    var col = row._thinkCol, target = row._contentTarget;
    if (col && col.parentNode) col.parentNode.replaceChild(target, col);
    row._thinkCol = null; row._thinkBlock = null; row._thinkToggle = null;
    row._thinkPanel = null; row._thinkState = null; row._thinkDots = null;
  }

  NGO.thinking = {
    // scrollFn：由 ChatCore 传入，用来把外层聊天窗口滚到底（可选）
    begin: function(row, scrollFn) {
      if (!row) return;
      if (!ensureBlock(row)) return;
      row._thinkScroll = scrollFn || null;
      var target = row._contentTarget;
      target.textContent = '';
      target._messageContent = null;
      var dots = document.createElement('span');
      dots.className = 'think-pending';
      // 点号本身也可以点开/收回，跟下方标记条等效
      dots.addEventListener('click', function(e) {
        e.stopPropagation();
        togglePanel(row);
      });
      target.appendChild(dots);
      row._thinkBlock.setAttribute('data-state', 'thinking');
      row._thinkToggle.textContent = '思考中';
      row._thinkUserToggled = false;
      row._thinkAutoCollapsed = false;
      row._thinkDots = dots;
      startDots(row, dots);
      // 思考中默认展开，生成回复时能直接看到思考过程
      setExpanded(row, true);
    },

    update: function(row, trace, identity) {
      if (!row || !row._thinkState) return;
      row._thinkState.trace = trace || [];
      row._thinkState.identity = identity;
      // 折叠状态下不渲染，等展开时再画，避免长思考白做 DOM
      if (row._thinkPanel && !row._thinkPanel.classList.contains('hidden')) {
        renderPanel(row);
        follow(row);
      }
    },

    // 回答正文开始出现：思考阶段结束，默认收起（用户手动展开过就不动）
    collapseForAnswer: function(row) {
      if (!row || !row._thinkState) return;
      removeDots(row);
      row._thinkDots = null;
      if (row._thinkUserToggled || row._thinkAutoCollapsed) return;
      row._thinkAutoCollapsed = true;
      setExpanded(row, false);
    },

    finish: function(row) {
      if (!row || !row._thinkState) return;
      removeDots(row);
      row._thinkDots = null;
      var hasTrace = row._thinkState.trace && row._thinkState.trace.length > 0;
      if (!hasTrace) { unwrap(row); return; }
      row._thinkBlock.setAttribute('data-state', 'done');
      row._thinkToggle.textContent = '思考过程';
      // 一帧回答都没流出来也要回到常态折叠
      if (!row._thinkUserToggled && !row._thinkAutoCollapsed) {
        row._thinkAutoCollapsed = true;
        setExpanded(row, false);
      }
    },

    // 历史回填：直接把这条已结束的思考链挂上去，折叠态、不播动画
    restore: function(row, trace, scrollFn) {
      if (!row || !trace || !trace.length) return false;
      if (!ensureBlock(row)) return false;
      row._thinkScroll = scrollFn || null;
      row._thinkState.trace = trace;
      row._thinkState.identity = null;
      row._thinkBlock.setAttribute('data-state', 'done');
      row._thinkToggle.textContent = '思考过程';
      row._thinkUserToggled = true;   // 历史行不参与"回答开始自动收起"
      row._thinkAutoCollapsed = true;
      setExpanded(row, false);
      return true;
    },

    // 行被丢弃（失败/取消/换存档）时调用，只做清理，不保留入口
    drop: function(row) {
      if (!row || !row._thinkState) return;
      removeDots(row);
      unwrap(row);
    },

    clearAll: function() {
      dotsEntries.forEach(function(entry) {
        if (entry.el && entry.el.parentNode) entry.el.parentNode.removeChild(entry.el);
      });
      dotsEntries = [];
      stopDots();
    },

    // 供测试与后续历史回填使用
    toggle: function(row) { return togglePanel(row); },
    isExpanded: function(row) {
      return !!(row && row._thinkPanel && !row._thinkPanel.classList.contains('hidden'));
    },
    traceOf: function(row) { return (row && row._thinkState) ? row._thinkState.trace : null; },
    DOTS: DOTS,
    TICK: TICK
  };
})();
