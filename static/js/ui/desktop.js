/**
 * view/desktop — 全局状态 + 桌面/任务栏图标事件
 */
'use strict';

var BASE = '.';
var NGO = window.NGO = window.NGO || {};
var showError = NGO.showError;
var apiJson = NGO.apiJson;
var playSfx = NGO.playSfx;

// State
var saveSelected = false;
var saveWindowFromDesktop = false;
var currentSaveSlot = null;
var saveSlotData = {1: false, 2: false, 3: false};
var pendingDeleteSlot = null;
var currentTopic = null;
var winZCounter = 50;

// DOM
var startMenu      = document.getElementById('start-menu');
var startBtn       = document.getElementById('start-btn');
var ytIcon         = document.getElementById('icon-youtube');
var otherIcons     = document.querySelectorAll('.desktop-icon.other');
var tooltip        = document.getElementById('tooltip');

// Desktop icon tooltips
var allIcons = document.querySelectorAll('.desktop-icon');
allIcons.forEach(function(icon) {
  icon.addEventListener('mouseenter', function() {
    var tip = icon.getAttribute('data-tooltip');
    if (tip && tooltip) {
      tooltip.textContent = tip;
      tooltip.style.display = 'block';
    }
  });
  icon.addEventListener('mouseleave', function() {
    if (tooltip) tooltip.style.display = 'none';
  });
  icon.addEventListener('mousemove', function(e) {
    if (tooltip && tooltip.style.display === 'block') {
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top  = (e.clientY + 12) + 'px';
    }
  });
});

// ── Taskbar Task Manager icon ──
var taskmgrIcon = document.getElementById('taskbar-taskmgr');
if (taskmgrIcon) {
  taskmgrIcon.addEventListener('click', function() {
    if (!saveSelected) return;
    var statsWin = document.getElementById('stats-window');
    if (!statsWin) return;
    if (statsWin.style.display === 'none') statsWin.style.display = '';
    if (NGO.getStats) NGO.getStats({ silent: true });
    NGO.WindowManager.focus(statsWin);
    playSfx('Click');
  });
}

// ── Taskbar JINE icon ──
var taskbarJine = document.getElementById('taskbar-jine');
if (taskbarJine) {
  taskbarJine.addEventListener('click', function() {
    if (!saveSelected) return;
    var jineWin = document.getElementById('jine-window');
    if (!jineWin) return;
    if (jineWin.style.display === 'none' || jineWin.classList.contains('hidden')) {
      jineWin.classList.remove('hidden');
      jineWin.style.display = '';
      updateJineDate();
    }
    NGO.WindowManager.focus(jineWin);
    playSfx('Click');
  });
}

// ── Desktop JINE icon ──
var jineIcon = document.getElementById('icon-jine');
if (jineIcon) {
  jineIcon.addEventListener('click', function() {
    if (!saveSelected) return;
    var jineWin = document.getElementById('jine-window');
    if (!jineWin) return;
    if (jineWin.style.display === 'none' || jineWin.classList.contains('hidden')) {
      jineWin.classList.remove('hidden');
      jineWin.style.display = '';
      updateJineDate();
    }
    NGO.WindowManager.focus(jineWin);
    playSfx('Click');
  });
}

// ── Desktop Help icon ──
var helpIcon = document.getElementById('icon-help');
if (helpIcon) {
  helpIcon.addEventListener('click', function() {
    if (!saveSelected) return;
    var win = document.getElementById('help-window');
    if (!win) return;
    if (win.style.display === 'none' || win.classList.contains('hidden')) {
      win.classList.remove('hidden');
      win.style.display = '';
    }
    NGO.WindowManager.focus(win);
    playSfx('Click');
  });
}

// ── Desktop YouTube icon ──
if (ytIcon) {
  ytIcon.addEventListener('click', function() {
    if (!saveSelected) {
      saveWindowFromDesktop = false;
      var saveWin = document.getElementById('save-window');
      if (saveWin) {
        saveWin.classList.remove('hidden');
        saveWin.style.setProperty('display', 'flex', 'important');
        NGO.WindowManager.focus(saveWin);
      }
      playSfx('Click');
      return;
    }
    var topicWin = document.getElementById('topic-window');
    if (topicWin) {
      topicWin.classList.remove('hidden');
      topicWin.style.display = '';
      NGO.WindowManager.focus(topicWin);
    }
    playSfx('Click');
  });
}
