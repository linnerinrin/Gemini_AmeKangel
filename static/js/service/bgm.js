/*
 services/bgm BGM 控制逻辑
 formatBGMTime //数字->时间
 updateProgress //播放进度
 startProgressTimer // 开始播放 计时器定期传进度
 stopProgressTimer // 停止 计时器
 setActiveBGMButton //dom 按钮样式
 getBGMDisplay // 曲名-文件映射
 scheduleBGMresume //慢启动
 playBGM //播放音乐和dom
 stopBGM //暂停音乐
 showBgmTooltip //悬浮栏
 hideHoverTooltip //悬浮栏

 */
'use strict';

var bgmAudio     = null;
var bgmPlaying   = false;
var currentBGM   = null;
var progressTimer = null;

function formatBGMTime(sec) {
  if (!isFinite(sec) || sec < 0) return '0:00';
  var m = Math.floor(sec / 60);
  var s = Math.floor(sec % 60);
  return m + ':' + (s < 10 ? '0' : '') + s;
}

function updateProgress() {
  if (!bgmAudio) return;
  var cur = bgmAudio.currentTime || 0;
  var dur = bgmAudio.duration || 0;
  var barFill = document.getElementById('psyche-bar-fill');
  var timeDisplay = document.getElementById('psyche-time');
  if (barFill) barFill.style.width = (dur > 0 ? (cur / dur) * 100 : 0) + '%';
  if (timeDisplay) timeDisplay.textContent = formatBGMTime(cur) + ' / ' + formatBGMTime(dur);
}

function startProgressTimer() { stopProgressTimer(); progressTimer = setInterval(updateProgress, 250); }
function stopProgressTimer() { if (progressTimer) { clearInterval(progressTimer); progressTimer = null; } }

function setActiveBGMButton(bgm) {
  document.querySelectorAll('.psyche-btn-wrap').forEach(function(b) {
    b.classList.remove('active');
    var btn = b.querySelector('.start-menu-btn');
    if (btn) btn.style.backgroundImage = '';
  });
  if (bgm) {
    var el = document.querySelector('.psyche-btn-wrap[data-bgm="' + bgm + '"]');
    if (el) { el.classList.add('active'); var btn = el.querySelector('.start-menu-btn'); if (btn) btn.style.backgroundImage = 'url(' + BASE + '/assets/images/BaseButtonPressed.png)'; }
  }
}

function getBGMDisplay(name) {
  var el = document.querySelector('.psyche-btn-wrap[data-bgm="' + name + '"]');
  if (el) { var span = el.querySelector('.start-menu-btn-text'); if (span) return span.textContent; }
  return name;
}

// 播放意图与实际播放状态分离；旧 Promise 不得覆盖新操作。
var _bgmPendingResume = null;
var _bgmGeneration = 0;
var bgmWantsPlayback = false;

function cancelBGMResume() {
  if (_bgmPendingResume) document.removeEventListener('click', _bgmPendingResume);
  _bgmPendingResume = null;
}

function renderBGMState() {
  var capsule = document.getElementById('psyche-capsule-btn');
  if (capsule) capsule.src = BASE + '/assets/images/icon_window_hiporon_capsule' + (bgmWantsPlayback ? '1' : '2') + '.png';
  setActiveBGMButton(currentBGM);
  ['psyche-now-playing', 'taskbar-bgm-name'].forEach(function(id) {
    var el = document.getElementById(id);
    if (el) el.textContent = currentBGM ? getBGMDisplay(currentBGM) : '';
  });
  updateProgress();
}

function scheduleBGMResume(audio, generation) {
  cancelBGMResume();
  function resume() {
    if (_bgmPendingResume !== resume) return;
    cancelBGMResume();
    if (audio !== bgmAudio || generation !== _bgmGeneration || !bgmWantsPlayback) return;
    attemptBGMPlay(audio, generation);
  }
  _bgmPendingResume = resume;
  document.addEventListener('click', resume);
}

function attemptBGMPlay(audio, generation) {
  function isCurrent() {
    return audio === bgmAudio && generation === _bgmGeneration && bgmWantsPlayback;
  }
  function failed(err) {
    if (!isCurrent()) return;
    bgmPlaying = false;
    stopProgressTimer();
    if (err && err.name === 'NotAllowedError') scheduleBGMResume(audio, generation);
    else bgmWantsPlayback = false;
    renderBGMState();
  }
  var result;
  try { result = audio.play(); } catch (err) { failed(err); return; }
  Promise.resolve(result).then(function() {
    if (!isCurrent()) {
      // 同一 Audio 已有更新的播放意图时，不得被旧回调暂停。
      if (audio !== bgmAudio || !bgmWantsPlayback) audio.pause();
      return;
    }
    cancelBGMResume();
    bgmPlaying = true;
    startProgressTimer();
    renderBGMState();
  }, failed);
}

function playBGM(name) {
  cancelBGMResume();
  var generation = ++_bgmGeneration;
  stopProgressTimer();
  bgmPlaying = false;
  bgmWantsPlayback = true;
  if (currentBGM !== name || !bgmAudio) {
    if (bgmAudio) bgmAudio.pause();
    bgmAudio = new Audio(BASE + '/assets/audio/' + name + '.wav');
    bgmAudio.volume = 0.5;
    bgmAudio.loop = true;
    currentBGM = name;
    var audio = bgmAudio;
    audio.addEventListener('loadedmetadata', function() {
      if (audio === bgmAudio) updateProgress();
    });
  }
  renderBGMState();
  attemptBGMPlay(bgmAudio, generation);
}

function stopBGM() {
  ++_bgmGeneration;
  bgmWantsPlayback = false;
  cancelBGMResume();
  if (bgmAudio) bgmAudio.pause();
  stopProgressTimer();
  bgmPlaying = false;
  // 保留实例、曲目和当前位置，暂停时也更新进度。
  renderBGMState();
}

function showBgmTooltip(el, name) {
  var hoverTooltip = document.getElementById('hover-tooltip');
  var hoverTitle = document.getElementById('hover-tooltip-title');
  var hoverSub = document.getElementById('hover-tooltip-sub');
  if (!hoverTooltip || !hoverTitle || !hoverSub) return;
  hoverTooltip.classList.remove('hidden'); hoverTooltip.classList.add('music-size');
  hoverTitle.textContent = name;
  var sub = el.getAttribute('data-sub'); hoverSub.textContent = sub || '';
  hoverTooltip.style.width = 'clamp(200px, 24vw, 300px)';
  hoverTooltip.style.height = sub ? 'clamp(100px, 14vh, 150px)' : 'clamp(72px, 10vh, 100px)';
  hoverTooltip.classList.add('show');
  var rect = el.getBoundingClientRect();
  var cx = rect.left + rect.width / 2; var cy = rect.top + rect.height / 2;
  var tw = hoverTooltip.offsetWidth; var th = hoverTooltip.offsetHeight;
  hoverTooltip.style.left = (cx - tw - 8) + 'px'; hoverTooltip.style.top = (cy - th - 8) + 'px';
}

function hideHoverTooltip() {
  var hoverTooltip = document.getElementById('hover-tooltip');
  if (hoverTooltip) { hoverTooltip.classList.remove('show'); hoverTooltip.classList.add('hidden'); }
}

NGO.showBgmTooltip = showBgmTooltip;
NGO.hideHoverTooltip = hideHoverTooltip;

NGO.setBgmVolume = function(v) {
  if (bgmAudio) bgmAudio.volume = v;
};
