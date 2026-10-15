/*
 * 通用文本/思考链渲染器（思考块本体在 service/thinking.js）
 */
(function() {
  'use strict';

  var NGO = window.NGO = window.NGO || {};

  function setTrustedTextAreaHtml(id, html) {
    var el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = html;
  }

  // identity 是 ChatCore 为每个请求创建的唯一对象；带 identity 的同一数组只允许追加。
  // 普通调用不信任数组身份，使用值快照验证完整前缀，支持整段替换和原地编辑。
  function thinkingKey(item) {
    return JSON.stringify([item.kind, item.kind === 'node' || item.kind === 'tool_call' ? item.name : (item.text || '')]);
  }

  function shouldFollow(el) {
    var selection = typeof window.getSelection === 'function' ? window.getSelection() : null;
    if (selection && !selection.isCollapsed) return false;
    return typeof el.clientHeight !== 'number' || el.scrollHeight - el.clientHeight - el.scrollTop <= 48;
  }

  function appendThinkingItem(el, item) {
    if (item.kind === 'node' || item.kind === 'tool_call') {
      var row = document.createElement('div'), label = document.createElement('strong');
      label.textContent = (item.kind === 'node' ? '节点: ' : '工具: ') + item.name;
      row.appendChild(label);
      el.appendChild(row);
      var group = document.createElement('div');
      group.className = 'think-node-group';
      el.appendChild(group);
      el._thinkGroup = group;
      return;
    }
    var span = document.createElement('span');
    span.textContent = item.text || '';
    if (span.style) span.style.whiteSpace = 'pre-wrap';
    (el._thinkGroup || el).appendChild(span);
  }

  NGO.renderThinking = function(el, value, identity) {
    var follow = shouldFollow(el);
    if (!Array.isArray(value)) {
      el.textContent = value || '';
      el._thinkState = null; el._thinkGroup = null;
      if (follow) el.scrollTop = el.scrollHeight;
      return;
    }
    var previous = el._thinkState;
    var count = previous ? previous.count : 0;
    var canAppend = !!previous && value.length >= count && previous.identity === identity;
    if (canAppend && identity) {
      canAppend = previous.source === value;
    } else if (canAppend) {
      // 没有受控流式 identity 的输入可能任意修改，必须逐项比对快照。
      for (var j = 0; j < count; j++) {
        if (previous.keys[j] !== thinkingKey(value[j])) { canAppend = false; break; }
      }
    }
    if (!canAppend) { el.textContent = ''; el._thinkGroup = null; count = 0; }
    for (var i = count; i < value.length; i++) appendThinkingItem(el, value[i]);
    el._thinkState = {identity: identity, source: value, count: value.length, keys: identity ? null : value.map(thinkingKey)};
    if (follow) el.scrollTop = el.scrollHeight;
  };

  // 思考链现在挂在回复行内（见 service/thinking.js），
  // 这里只保留通用渲染器：谁拿到容器谁调用。

  setTrustedTextAreaHtml('help-text-area',
    '<b>帮助文档</b><br><br>' +
    '<b>基本操作：</b><br>' +
    '- 点击桌面图标打开窗口<br>' +
    '- 拖动标题栏移动窗口<br>' +
    '- 点击窗口右上角X关闭窗口<br>' +
    '- 点击任务栏图标切换窗口<br>' +
    '- 点击"直播"选择直播主题<br><br>' +
    '<b>直播：</b><br>' +
    '- 在右侧面板发送弹幕<br>' +
    '- 点击表情按钮发送表情<br>' +
    '- 不同主题有不同角色和对话<br><br>' +
    '<b>状态查看：</b><br>' +
    '- Task Manager 显示实时数据<br>' +
    '- 状态影响角色动画和表情<br><br>' +
    '<b>快捷键：</b><br>' +
    '- 暂无'
  );
})();
