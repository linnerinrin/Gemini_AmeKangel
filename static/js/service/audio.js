/**
NGO.playSfx(name) name.wav导入进对象
NGO.setSfxVolume 设置音量

 */
(function() {
  'use strict';

  var BASE = '.';
  var NGO = window.NGO = window.NGO || {};
  var sounds = {};
  var sfxVolume = 0.4;

  NGO.playSfx = function(name) {
    try {
      var key = BASE + '/assets/audio/' + name + '.wav';
      var audio = sounds[key];
      if (!audio) {
        audio = new Audio(key);
        sounds[key] = audio;
      }
      audio.volume = sfxVolume; //音量设置
      audio.currentTime = 0;
      audio.play().catch(function() {}); //交互前播放音频会抛Dom expection catch一下
    } catch (e) {}
  };

  NGO.setSfxVolume = function(value) {
    sfxVolume = Math.max(0, Math.min(1, value));
  };

})();
