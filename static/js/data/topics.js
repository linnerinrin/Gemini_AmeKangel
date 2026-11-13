/*
直播 选项
 */
(function() {
  'use strict';

  var NGO = window.NGO = window.NGO || {};

  // mode: 每个直播主题对应后端一个独立 mode，保证各自的上下文/记忆互不串台
  //      0 = jine(糖糖私聊)  1 = 超天酱直播  2 = 堕天酱直播  3 = 糖糖直播
  NGO.topicConfigs = [
    { id: 0, title: "超天酱的直播", sub: "天使的降临", netachip: "netachip_001.png", charVariant: "kangel", mode: "1", actionMap: { idle: "idle", think: "think", smile: "smile", shobon: "shobon", start: "start" }, angelIntro: true, locked: false },
    { id: 1, title: "堕天酱直播", sub: "天使的堕落", netachip: "netachip_001.png", charVariant: "dangel", mode: "2", actionMap: { idle: "idle", think: "think", smile: "smile", shobon: "shobon", start: "start" }, angelIntro: true, locked: false },
    { id: 2, title: "糖糖的直播", sub: "天使以真面目降临", netachip: "netachip_001.png", charVariant: "ame", mode: "3", actionMap: { idle: "idle", think: "think", smile: "smile", shobon: "shobon", start: "start" }, angelIntro: false, locked: false },
    { id: 3, title: "???", sub: "未知", netachip: "netachip_alpha_unknown.png", charVariant: "", mode: "", actionMap: {}, angelIntro: false, locked: true },
    { id: 4, title: "???", sub: "未知", netachip: "netachip_alpha_unknown.png", charVariant: "", mode: "", actionMap: {}, angelIntro: false, locked: true },
    { id: 5, title: "???", sub: "未知", netachip: "netachip_alpha_unknown.png", charVariant: "", mode: "", actionMap: {}, angelIntro: false, locked: true },
    { id: 6, title: "???", sub: "未知", netachip: "netachip_alpha_unknown.png", charVariant: "", mode: "", actionMap: {}, angelIntro: false, locked: true },
    { id: 7, title: "???", sub: "未知", netachip: "netachip_alpha_unknown.png", charVariant: "", mode: "", actionMap: {}, angelIntro: false, locked: true },
    { id: 8, title: "???", sub: "未知", netachip: "netachip_alpha_unknown.png", charVariant: "", mode: "", actionMap: {}, angelIntro: false, locked: true },
    { id: 9, title: "???", sub: "未知", netachip: "netachip_alpha_unknown.png", charVariant: "", mode: "", actionMap: {}, angelIntro: false, locked: true },
    { id: 10, title: "???", sub: "未知", netachip: "netachip_alpha_unknown.png", charVariant: "", mode: "", actionMap: {}, angelIntro: false, locked: true },
    { id: 11, title: "???", sub: "未知", netachip: "netachip_alpha_unknown.png", charVariant: "", mode: "", actionMap: {}, angelIntro: false, locked: true }
  ];
})();
