/**
 * ui/bgm-menu — 开始菜单 + BGM 菜单
 */
'use strict';

// ═══ 开始菜单 ═══

var startBtnEl  = document.getElementById('start-btn');
var startMenuEl = document.getElementById('start-menu');

if (startBtnEl && startMenuEl) {
  startBtnEl.addEventListener('click', function(e) { e.stopPropagation(); startMenuEl.classList.toggle('open'); playSfx('Click'); });
  document.addEventListener('click', function() { startMenuEl.classList.remove('open'); if (psycheMenu) psycheMenu.classList.remove('open'); });
  startMenuEl.addEventListener('click', function(e) { e.stopPropagation(); });
}

var smControl = document.getElementById('sm-control');
if (smControl) {
  smControl.addEventListener('click', function(e) {
    e.stopPropagation(); startMenuEl.classList.remove('open');
    var win = document.getElementById('control-window');
    if (win) { win.classList.remove('hidden'); win.style.display = ''; }
    if (win) NGO.WindowManager.focus(win);
    playSfx('Click');
  });
}

var smSave = document.getElementById('sm-save');
if (smSave) {
  smSave.addEventListener('click', function(e) {
    e.stopPropagation(); startMenuEl.classList.remove('open');
    if (!saveSelected) return;
    saveWindowFromDesktop = true;
    if (typeof loadSaveSlots === 'function') loadSaveSlots();
    var saveWin = document.getElementById('save-window');
    if (saveWin) { saveWin.classList.remove('hidden'); saveWin.style.setProperty('display', 'flex', 'important'); NGO.WindowManager.focus(saveWin); }
    playSfx('Click');
  });
}

var smLogout = document.getElementById('sm-logout');
if (smLogout) {
  smLogout.addEventListener('click', function(e) {
    e.stopPropagation(); startMenuEl.classList.remove('open');
    if (!saveSelected) return;
    var logoutWin = document.getElementById('logout-window');
    if (logoutWin) { logoutWin.classList.remove('hidden'); logoutWin.style.setProperty('display', 'flex', 'important'); NGO.WindowManager.focus(logoutWin); }
    playSfx('Click');
  });
}

// ═══ BGM 菜单 (Psyche) ═══

var psycheBtn  = document.getElementById('psyche-btn');
var psycheMenu = document.getElementById('psyche-menu');

if (psycheBtn && psycheMenu) {
  psycheBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    if (!saveSelected) { playSfx('Click'); return; }
    var opening = !psycheMenu.classList.contains('open');
    psycheMenu.classList.toggle('open');
    startMenuEl.classList.remove('open');
    if (opening && !bgmPlaying && !currentBGM) playBGM('mainloop_normal');
    playSfx('Click');
  });
  psycheMenu.addEventListener('click', function(e) { e.stopPropagation(); });
}

var capsuleBtn = document.getElementById('psyche-capsule-btn');
if (capsuleBtn) {
  capsuleBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    if (bgmWantsPlayback) stopBGM();
    else if (currentBGM) playBGM(currentBGM);
    else playBGM('mainloop_normal');
  });
}

var psycheBtns = document.querySelectorAll('.psyche-btn-wrap');
psycheBtns.forEach(function(btn) {
  btn.addEventListener('click', function(e) {
    e.stopPropagation();
    var bgm = btn.getAttribute('data-bgm');
    if (currentBGM === bgm && bgmWantsPlayback) stopBGM(); else playBGM(bgm);
  });
  var bgmName = btn.getAttribute('data-bgm');
  var displayName = btn.querySelector('.start-menu-btn-text');
  var label = displayName ? displayName.textContent : bgmName;
  btn.addEventListener('mouseenter', function(e) { NGO.showBgmTooltip(btn, label); });
  btn.addEventListener('mouseleave', NGO.hideHoverTooltip);
});
