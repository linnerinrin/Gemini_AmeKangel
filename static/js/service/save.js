/*
 services/save-manager 存档编排
 update saveslot dom 初始化单slot{avatar选择 trash删掉}
 register saveslot
 unregister saveslot
 load saveslot 调api 拿session_id
 load sessions history 拿session_history
 enter saveslot 进桌面
 create and enter saveslot
 choose and enter saveslot
 finish delete saveslot

 */
'use strict';

function updateSaveSlotDOM(slotNum) {
  var slot = document.querySelector('.save-slot[data-slot="' + slotNum + '"]');
  if (!slot) return;
  var avatar = slot.querySelector('.save-slot-avatar');
  var trash = slot.querySelector('.save-slot-trash');
  if (saveSlotData[slotNum]) {
    slot.classList.remove('unregistered'); slot.classList.add('registered');
    if (avatar) avatar.src = BASE + '/assets/images/icon_cho.png';
    if (trash) trash.classList.remove('hidden');
  } else {
    slot.classList.remove('registered'); slot.classList.add('unregistered');
    if (avatar) avatar.src = BASE + '/assets/images/icon_ame.png';
    if (trash) trash.classList.add('hidden');
  }
}

function registerSaveSlot(slotNum) { saveSlotData[slotNum] = true; updateSaveSlotDOM(slotNum); }
function unregisterSaveSlot(slotNum) { saveSlotData[slotNum] = false; updateSaveSlotDOM(slotNum); }

function loadSaveSlots(callback) {
  NGO.saveAPI.getSessions().then(function(res) {
    saveSlotData = {1: false, 2: false, 3: false};
    (res.sessions || []).forEach(function(item) {
      var n = parseInt(item.session_num, 10);
      if (n >= 1 && n <= 3) saveSlotData[n] = true;
    });
    [1, 2, 3].forEach(updateSaveSlotDOM);
    if (callback) callback();
  }).catch(function(err) { showError(err.message || '存档列表加载失败'); if (callback) callback(); });
}

function loadSessionHistory() {
  var generation = NGO.chatSessionGeneration || 0;
  NGO.jineHistoryLoading = true;
  NGO.chatAPI.getHistory('0').then(function(history) {
    if (generation !== (NGO.chatSessionGeneration || 0)) return;
    if (NGO.jineChat) NGO.jineChat.loadHistory(history || []);
  }).catch(function(err) {
    if (generation === (NGO.chatSessionGeneration || 0)) showError(err.message || '历史会话加载失败');
  }).finally(function() {
    if (generation === (NGO.chatSessionGeneration || 0)) NGO.jineHistoryLoading = false;
  });
}

function resetDesktopState() {
  NGO.chatSessionGeneration = (NGO.chatSessionGeneration || 0) + 1;
  if (NGO.activeChatController) NGO.activeChatController.abort();
  if (typeof stopAmeAnimation === 'function') stopAmeAnimation();
  if (typeof stopLiveAnimation === 'function') stopLiveAnimation();

  // 窗口：全隐藏 + 清掉拖拽产生的 inline 位置
  document.querySelectorAll('.window').forEach(function(w) {
    w.classList.add('hidden');
    w.style.setProperty('display', 'none', 'important');
    w.style.left = ''; w.style.top = ''; w.style.right = ''; w.style.bottom = ''; w.style.transform = '';
  });

  // BGM：切换存档时停止播放，但保留当前曲目和暂停位置
  if (typeof stopBGM === 'function') stopBGM();

  // 文本 / 弹幕
  if (NGO.thinking) NGO.thinking.clearAll();
  var liveReply = document.getElementById('live-reply-overlay');
  if (liveReply) liveReply.textContent = '';
  clearDanmakuDisplay();

  // 状态变量
  currentTopic = null;
  saveSelected = false;
}

function enterSaveSlot(slotNum) {
  saveSelected = true;
  currentSaveSlot = slotNum;
  document.querySelectorAll('.save-slot').forEach(function(s) { s.classList.remove('active'); });
  var slot = document.querySelector('.save-slot[data-slot="' + slotNum + '"]');
  if (slot) slot.classList.add('active');
  var saveWin = document.getElementById('save-window');
  if (saveWin) { saveWin.classList.add('hidden'); saveWin.style.setProperty('display', 'none', 'important'); }

  resetDesktopState();
  saveSelected = true;  // resetDesktopState 设成了 false，这里恢复

  revealDesktop(); loadSessionHistory();
  if (NGO.getStats) NGO.getStats({ silent: true });
  playSfx('Click');
}

function createAndEnterSaveSlot(slotNum) {
  NGO.saveAPI.newSession(slotNum).then(function(res) {
    if (!res.ok) { showError(res.msg || '存档创建失败'); return; }
    registerSaveSlot(slotNum); enterSaveSlot(slotNum);
  }).catch(function(err) { showError(err.message || '存档创建失败'); });
}

function chooseAndEnterSaveSlot(slotNum) {
  NGO.saveAPI.chooseSession(slotNum).then(function(res) {
    if (!res.ok) { showError(res.msg || '存档会话启动失败'); return; }
    enterSaveSlot(slotNum);
  }).catch(function(err) { showError(err.message || '存档会话启动失败'); });
}

function finishDeleteSaveSlot(slotNum) {
  unregisterSaveSlot(slotNum);
  if (currentSaveSlot === slotNum) {
    currentSaveSlot = null;
    resetDesktopState();
    document.querySelectorAll('.desktop-icon.other').forEach(function(icon) { icon.classList.add('hidden'); });
    var ytLabel = document.getElementById('yt-label');
    if (ytLabel) ytLabel.textContent = '直播';
    saveWindowFromDesktop = false;
    var saveWin = document.getElementById('save-window');
    if (saveWin) { saveWin.classList.remove('hidden'); saveWin.style.setProperty('display', 'flex', 'important'); }
  }
}

window.loadSaveSlots = loadSaveSlots;

// ── Save slot click ──
var saveSlots = document.querySelectorAll('.save-slot');
saveSlots.forEach(function(slot) {
  slot.addEventListener('click', function(e) {
    if (e.target.closest('.save-slot-trash')) return;
    e.stopPropagation();
    var slotNum = parseInt(slot.getAttribute('data-slot'));
    if (!saveSlotData[slotNum]) {
      createAndEnterSaveSlot(slotNum);
      return;
    }
    chooseAndEnterSaveSlot(slotNum);
  });

  var trash = slot.querySelector('.save-slot-trash');
  if (trash) {
    trash.addEventListener('click', function(e) {
      e.stopPropagation();
      pendingDeleteSlot = parseInt(slot.getAttribute('data-slot'));
      var label = document.getElementById('delete-save-label');
      if (label) label.textContent = '确定删除存档 ' + pendingDeleteSlot + ' 吗？';
      var win = document.getElementById('delete-save-window');
      if (win) {
        win.classList.remove('hidden');
        win.style.display = '';
        winZCounter++;
        win.style.zIndex = winZCounter;
      }
      playSfx('Click');
    });
  }
});

// ── Delete confirm / cancel ──
var deleteConfirm = document.getElementById('delete-confirm');
var deleteCancel = document.getElementById('delete-cancel');
var deleteWin = document.getElementById('delete-save-window');

if (deleteConfirm && deleteWin) {
  deleteConfirm.addEventListener('click', function(e) {
    e.stopPropagation();
    var slotNum = pendingDeleteSlot;
    if (slotNum) {
      NGO.saveAPI.deleteSession(slotNum).then(function() {
        finishDeleteSaveSlot(slotNum);
      }).catch(function(err) {
        showError(err.message || '存档删除失败');
      });
      pendingDeleteSlot = null;
      deleteWin.classList.add('hidden');
      deleteWin.style.setProperty('display', 'none', 'important');
      playSfx('Click');
      return;
    }
    deleteWin.classList.add('hidden');
    deleteWin.style.setProperty('display', 'none', 'important');
    playSfx('Click');
  });
}

if (deleteCancel && deleteWin) {
  deleteCancel.addEventListener('click', function(e) {
    e.stopPropagation();
    pendingDeleteSlot = null;
    deleteWin.classList.add('hidden');
    deleteWin.style.setProperty('display', 'none', 'important');
    playSfx('Click');
  });
}
