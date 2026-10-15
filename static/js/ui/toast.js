/**
NGO.showError 报错时show  报错提示框
 */
(function() {
  'use strict';

  var NGO = window.NGO = window.NGO || {};
  var errorToast = document.getElementById('error-toast');
  var errorToastText = document.getElementById('error-toast-text');
  var errorToastTimer = null;

  NGO.showError = function(msg) {
    if (!errorToast || !errorToastText) return;
    errorToastText.textContent = msg;
    errorToast.classList.remove('hidden');
    errorToast.classList.add('show');
    if (errorToastTimer) clearTimeout(errorToastTimer);
    errorToastTimer = setTimeout(function() {
      errorToast.classList.remove('show');
      setTimeout(function() { errorToast.classList.add('hidden'); }, 300);
    }, 2500);
  };
})();
