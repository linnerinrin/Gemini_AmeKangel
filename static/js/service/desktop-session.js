/*
 services/desktop-session 存档桌面生命周期
 */
'use strict';

function revealDesktop() {
  document.querySelectorAll('.desktop-icon.other').forEach(function(icon) { icon.classList.remove('hidden'); });
  var ytLabel = document.getElementById('yt-label');
  if (ytLabel) ytLabel.textContent = '直播选择';
  ['ame-window','jine-window','stats-window'].forEach(function(id) {
    var w = document.getElementById(id); if (w) { w.classList.remove('hidden'); w.style.display = ''; }
  });
  startAmeAnimation();
  var autoplay = document.getElementById('bgm-autoplay');
  if (typeof playBGM === 'function' && !currentBGM && (!autoplay || autoplay.checked)) playBGM('mainloop_normal');
}

var logoutConfirm = document.getElementById('logout-confirm');
var logoutCancel  = document.getElementById('logout-cancel');
var logoutWin     = document.getElementById('logout-window');

if (logoutConfirm && logoutWin) {
  logoutConfirm.addEventListener('click', function(e) {
    e.stopPropagation();
    logoutWin.classList.add('hidden');
    logoutWin.style.setProperty('display', 'none', 'important');
    resetDesktopState();
    currentSaveSlot = null;
    document.querySelectorAll('.desktop-icon.other').forEach(function(icon) { icon.classList.add('hidden'); });
    saveWindowFromDesktop = false;
    if (typeof loadSaveSlots === 'function') loadSaveSlots();
    var saveWin = document.getElementById('save-window');
    if (saveWin) { saveWin.classList.remove('hidden'); saveWin.style.setProperty('display', 'flex', 'important'); }
    playSfx('Click');
  });
}

if (logoutCancel && logoutWin) {
  logoutCancel.addEventListener('click', function(e) {
    e.stopPropagation();
    logoutWin.classList.add('hidden');
    logoutWin.style.setProperty('display', 'none', 'important');
    playSfx('Click');
  });
}

loadSaveSlots();
