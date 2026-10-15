/*
   全局时钟
   gettimeperiod //时间->时段
   updateclockicon //时段->时钟样式
   updateclock //更新时间
   formatdatetime //当前时间
   updatetimeof day //由时间确立背景
 */
(function() {
  'use strict';

  var BASE = '.';
  var barLeft = document.getElementById('side-bar-left');
  var barRight = document.getElementById('side-bar-right');

  function getTimePeriod(h) {
    if (h >= 6 && h < 17) return 'noon';
    if (h >= 17 && h < 20) return 'evening';
    return 'night';
  }

  function updateClockIcon(h) {
    var icon = document.getElementById('clock-icon');
    if (!icon) return;
    var period = getTimePeriod(h);
    icon.src = BASE + '/assets/images/icon_watch_' + period + '.png';
  }

  window.NGO = window.NGO || {};

  function updateClock() {
    var now = Date.now();
    NGO.now = now;
    var clockText = document.getElementById('clock-text');
    if (clockText) {
      var d = new Date(now);
      var h = d.getHours();
      var m = d.getMinutes();
      clockText.textContent =
        (h < 10 ? '0' : '') + h + ':' + (m < 10 ? '0' : '') + m;
      updateClockIcon(h);
      updateTimeOfDay(h);
    }
    var jineDate = document.querySelector('.jine-date-text');
    if (jineDate) jineDate.textContent = NGO.formatDateTime();
  }

  NGO.formatDateTime = function() {
    var d = new Date(NGO.now);
    var y = d.getFullYear();
    var mo = String(d.getMonth() + 1).padStart(2, '0');
    var day = String(d.getDate()).padStart(2, '0');
    var h = String(d.getHours()).padStart(2, '0');
    var mi = String(d.getMinutes()).padStart(2, '0');
    return y + '/' + mo + '/' + day + ' ' + h + ':' + mi;
  };

  function updateTimeOfDay(h) {
    if (h === undefined) h = new Date().getHours();
    var bg;
    if (h >= 6 && h < 17) {
      bg = 'bg_side_bar_noon2.png';
    } else if (h >= 17 && h < 20) {
      bg = 'bg_side_bar_evening2.png';
    } else {
      bg = 'bg_side_bar_night2.png';
    }
    var url = BASE + '/assets/images/' + bg;
    if (barLeft) barLeft.style.backgroundImage = 'url(' + url + ')';
    if (barRight) barRight.style.backgroundImage = 'url(' + url + ')';
  }

  updateClock();
  setInterval(updateClock, 30000);
})();
