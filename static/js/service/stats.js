/*
services/stats status
  setStat 传入status
  getStats 从后端拿全量status
 */
'use strict';

NGO.status = { followers: 0, stress: 0, affection: 50, mental: 0 };
NGO._statHandlers = [];

// 与 animation.js 共用一份钩子列表，避免动画与数值变化各走一套事件。
function onStatusChange(handler) {
  if (typeof handler !== 'function') throw new TypeError('status handler must be a function');
  NGO._statHandlers.push(handler);
  return function() {
    var i = NGO._statHandlers.indexOf(handler);
    if (i !== -1) NGO._statHandlers.splice(i, 1);
  };
}

function setStat(name, val, opts) {
  var old = NGO.status[name];
  NGO.status[name] = val;
  var cores = document.querySelectorAll('.stats-core[data-stat="' + name + '"]');
  for (var i = 0; i < cores.length; i++) {
    cores[i].textContent = name === 'followers' ? Number(val).toLocaleString() : val;
  }
  if (name !== 'followers') {
    var fills = document.querySelectorAll('.stats-bar-fill[data-stat="' + name + '"]');
    for (var j = 0; j < fills.length; j++) fills[j].style.height = Math.min(100, Math.max(0, val)) + '%';
  }
  // 即使当前没有数值 DOM，也不能吞掉状态变化钩子。
  for (var k = 0; k < NGO._statHandlers.length; k++) {
    NGO._statHandlers[k](name, old, val, opts || {});
  }
}

setStat('followers', 0); setStat('stress', 0); setStat('affection', 50); setStat('mental', 0);

function getStats(opts) {
  if (!saveSelected) return;
  NGO.get_status.getStatus().then(function(status) {
    if (!status) return;
    if (status.followers != null) setStat('followers', status.followers, opts);
    if (status.stress != null) setStat('stress', status.stress, opts);
    if (status.affection != null) setStat('affection', status.affection, opts);
    if (status.mental != null) setStat('mental', status.mental, opts);
  }).catch(function() {});
}

window.NGO = window.NGO || {};
window.NGO.setStat = setStat;
window.NGO.onStatusChange = onStatusChange;
window.NGO.isSaveSelected = function() { return saveSelected; };
window.NGO.getStats = getStats;
