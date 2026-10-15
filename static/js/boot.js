/**
 * NEEDY GIRL OVERDOSE - Boot Sequence
 */
(function() {
  'use strict';

  var BASE = '/static';

  // Pre-create and hold audio references to prevent GC
  var _sounds = {};

  function playSfx(name) {
    try {
      var key = BASE + '/assets/audio/' + name + '.wav';
      var a = _sounds[key];
      if (!a) {
        a = new Audio(key);
        a.volume = 0.5;
        _sounds[key] = a;
      }
      a.currentTime = 0;
      a.play().catch(function() {});
    } catch (e) {}
  }

  var biosLayer   = document.getElementById('bios-layer');
  var logoLayer   = document.getElementById('logo-layer');
  var logoArea    = document.getElementById('bios-logo-area');

  var biosDone  = false;
  var logoShown = false;
  var dismissed = false;

  var SEQ = [
    { t: 100,  id: 'bios-0', cls: 'visible' },
    { t: 400,  id: 'bios-1', cls: 'visible' },
    { t: 800,  id: 'bios-3', cls: 'visible' },
    { t: 1100, id: 'bios-4', cls: 'visible' },
    { t: 1500, id: 'bios-6', cls: 'visible' },
    { t: 1800, id: 'bios-7', cls: 'visible' },
    { t: 2100, id: 'bios-8', cls: 'visible' },
    { t: 2400, id: 'bios-9', cls: 'visible' },
    { t: 2800, id: 'bios-11', cls: 'visible' },
    { t: 3100, id: 'bios-12', cls: 'visible' },
    { t: 3400, id: 'bios-13', cls: 'visible' },
    { t: 3700, id: 'bios-14', cls: 'visible' },
    { t: 4100, fn: function() { logoArea.classList.add('show'); playSfx('BIOS_piko'); } },
    { t: 4500, id: 'bios-17', cls: 'visible' },
    { t: 5000, id: 'bios-19', cls: 'visible' },
    { t: 5300, id: 'bios-20', cls: 'visible' },
    { t: 5700, id: 'bios-22', cls: 'visible' },
    { t: 6200, fn: function() { playSfx('Boot_Caution'); } },
    {
      t: 6500,
      fn: function() {
        biosDone = true;
        biosLayer.style.opacity = '0';
        biosLayer.style.transition = 'opacity 1s';
        setTimeout(function() {
          logoShown = true;
          logoLayer.classList.add('show');
          playSfx('game_start');
        }, 1000);
      }
    }
  ];

  var timers = [];

  SEQ.forEach(function(item) {
    var timer = setTimeout(function() {
      if (item.id) {
        var el = document.getElementById(item.id);
        if (el) el.classList.add(item.cls);
      }
      if (item.fn) item.fn();
    }, item.t);
    timers.push(timer);
  });

  function skipToLogo() {
    timers.forEach(clearTimeout);
    biosDone = true;
    if (biosLayer) {
      biosLayer.style.opacity = '0';
      biosLayer.style.transition = 'opacity 0.3s';
    }
    setTimeout(function() {
      if (!logoShown) {
        logoShown = true;
        logoLayer.classList.add('show');
        playSfx('game_start');
      }
    }, 300);
  }

  function dismiss() {
    if (dismissed) return;
    dismissed = true;
    playSfx('Log_In');
    setTimeout(function() {
      var appOrigin = window.location.port === '63342'
        ? window.location.protocol + '//localhost:5577'
        : window.location.origin;
      window.location.href = appOrigin + '/static/desktop.html';
    }, 300);
  }

  function handleClick(e) {
    playSfx('Click');
    if (!biosDone) {
      skipToLogo();
    } else if (logoShown) {
      dismiss();
    }
  }

  function handleKey(e) {
    if (e.key === 'F5' || e.key === 'F12') return;
    if (e.key !== 'Enter' && e.key !== ' ' && e.key !== 'Spacebar') return;
    e.preventDefault();
    handleClick(e);
  }

  document.addEventListener('click', handleClick);
  document.addEventListener('keydown', handleKey);

})();
