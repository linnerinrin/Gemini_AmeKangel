/**
 * NEEDY GIRL OVERDOSE - Window Manager
 */
'use strict';

window.NGO = window.NGO || {};

var registeredWindows = {};

// ── Window focus ──
document.addEventListener('mousedown', function(e) {
  var win = e.target.closest('.window');
  if (!win) {
    document.querySelectorAll('.window').forEach(function(w) {
      w.classList.remove('active');
    });
    return;
  }
  if (!win.classList.contains('active')) {
    document.querySelectorAll('.window').forEach(function(w) {
      w.classList.remove('active');
    });
    win.classList.add('active');
  }
  winZCounter++;
  win.style.zIndex = winZCounter;
  var cfg = registeredWindows[win.id];
  if (cfg && cfg.onFocus) cfg.onFocus(win);
});

// ── Window dragging ──
var dragTarget = null;
var dragOffX = 0;
var dragOffY = 0;

document.addEventListener('mousedown', function(e) {
  var header = e.target.closest('.window-header');
  if (!header) return;
  var win = header.closest('.window');
  if (!win) return;
  var cfg = registeredWindows[win.id];
  if (cfg && cfg.draggable === false) return;
  dragTarget = win;
  var rect = win.getBoundingClientRect();
  dragOffX = e.clientX - rect.left;
  dragOffY = e.clientY - rect.top;
  win.style.left = (rect.left / window.innerWidth * 100) + '%';
  win.style.top = (rect.top / window.innerHeight * 100) + '%';
  win.style.right = 'auto';
  win.style.bottom = 'auto';
  win.style.transform = 'none';
});

document.addEventListener('mousemove', function(e) {
  if (!dragTarget) return;
  var leftPct = ((e.clientX - dragOffX) / window.innerWidth) * 100;
  var topPct = ((e.clientY - dragOffY) / window.innerHeight) * 100;
  dragTarget.style.left = leftPct + '%';
  dragTarget.style.top  = topPct + '%';
  dragTarget.style.right = 'auto';
  dragTarget.style.bottom = 'auto';
});

document.addEventListener('mouseup', function() {
  if (dragTarget) {
    dragTarget = null;
  }
});

// ── Close buttons ──
document.addEventListener('click', function(e) {
  var closeBtn = e.target.closest('.win-close');
  if (!closeBtn) return;
  if (closeBtn.classList.contains('hidden')) return;
  var win = closeBtn.closest('.window');
  if (!win) return;
  var cfg = registeredWindows[win.id] || {};
  var closable = typeof cfg.closable === 'function' ? cfg.closable(win) : cfg.closable;
  if (closable === false) return;
  win.classList.add('hidden');
  win.style.setProperty('display', 'none', 'important');
  win.classList.remove('active');
  if (cfg.onClose) cfg.onClose(win);
  playSfx('Click');
});

// ── Public API ──
NGO.WindowManager = {
  register: function(el, options) {
    if (!el || !el.id) return;
    registeredWindows[el.id] = options || {};
  },

  focus: function(el) {
    if (!el) return;
    document.querySelectorAll('.window').forEach(function(w) { w.classList.remove('active'); });
    el.classList.add('active');
    winZCounter++;
    el.style.zIndex = winZCounter;
    var cfg = registeredWindows[el.id];
    if (cfg && cfg.onFocus) cfg.onFocus(el);
  },

  revealAll: function() {
    for (var id in registeredWindows) {
      var win = document.getElementById(id);
      if (win) { win.classList.remove('hidden'); win.style.display = ''; }
    }
  }
};

NGO.WindowManager.register(document.getElementById('help-window'), {});
