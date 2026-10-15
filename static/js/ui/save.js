/**
 * ui/save — 存档窗口 + 删除存档确认窗口
 */
'use strict';

window.NGO = window.NGO || {};

NGO.WindowManager.register(document.getElementById('save-window'), {
  onClose: function() {
    saveWindowFromDesktop = false;
  }
});

NGO.WindowManager.register(document.getElementById('delete-save-window'), {
  onClose: function() {
    pendingDeleteSlot = null;
  }
});
