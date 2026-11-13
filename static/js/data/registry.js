/**
 * services/registry — 所有配置集中管理
 * 纯配置数据，逻辑由各模块实现
 */
'use strict';

window.NGO = window.NGO || {};

// ═══════════════════════════════════════
// 表情标签（与 desktop.html 的 emoji 弹窗按钮、prompts/*_sys.txt 保持一致）
// ═══════════════════════════════════════
NGO.emojiCodes = [
  'jine_pien_compromise', 'jine_pien_die', 'jine_pien_loveforever', 'jine_pien_nyan',
  'jine_pien_ok', 'jine_pien_praise', 'jine_pien_sorry', 'jine_pien_uncare'
];

// 中文标签 -> 代码：模型偶尔会输出 <永远爱你> 这类标签，历史里也可能存着旧数据，
// 这里统一兜底成代码，保证聊天区、历史回填与直播字幕都能渲染成表情图片。
NGO.emojiAliases = {
  '你说得对': 'jine_pien_compromise',
  '死了': 'jine_pien_die',
  '永远爱你': 'jine_pien_loveforever',
  '嘤嘤': 'jine_pien_nyan',
  '好': 'jine_pien_ok',
  '太强了': 'jine_pien_praise',
  '抱歉': 'jine_pien_sorry',
  '关我屁事': 'jine_pien_uncare'
};

// ═══════════════════════════════════════
// BGM 曲目注册（每个 wav 一个条目，与 psyche menu HTML 保持同步）
// ═══════════════════════════════════════
var bgmTracks = {
  'mainloop_normal':              { sub: '与糖糖的轻松日常' },
  'mainloop_middle':              { sub: '糖糖的斗争陷入白热化' },
  'mainloop_yami':                { sub: '与糖糖的轻松日常……吗？' },
  'mainloop_gedarranged':         { sub: '糖糖似乎有点危险' },
  'mainloop_emoarranged':         { sub: '天使也是会流泪的' },
  'mainloop_normaledarranged':    { sub: '千万数据的大冲击' },
  'mainloop_happyarranged':       { sub: '世界虽苦，有你皆甜' },
  'mainloop_shuban':              { sub: '好像有不好的事情要发生' },
  'event_yami':                   { sub: '因为，最痛的不是手腕' },
  'event_happy':                  { sub: '普通的生活对她而言是完美的吗' },
  'event_emo':                    { sub: '我爱你爱你爱你爱你爱你一千遍' },
  'ending_grand':                 { sub: '刚才谁给你发微信？' },
  'ending_dead':                  { sub: '天使的坠落' },
  'ending_ginga':                 { sub: '应许之地便是那永远的乌托邦' },
  'ending_ideon':                 { sub: '至少曾经努力过，不是吗' },
  'InternetOverdose':             { sub: '点赞，点踩，对你来说是致命毒药' },
  'internetoverdose_offvocal':    { sub: '点赞，点踩，对你来说是致命毒药' },
  'INTERNET OVERDOSE8bit':        { sub: '点赞，点踩，对你来说是致命毒药' },
  'INTERNET YAMERO':              { sub: '这垃圾互联网待不下去了，我们一起逃走去看海吧' },
  'INTERNET YAMERO_offvocal':     { sub: '这垃圾互联网待不下去了，我们一起逃走去看海吧' },
  '03 月虹蝶':                    { sub: '「月神一直守护着愚不可及的二人」' },
  '04 月虹蝶(inst)':              { sub: '「月神一直守护着愚不可及的二人」' },
  'きゅびずむ':                   { sub: '裁剪又贴上的皮，看见开裂的疤' },
  'きゅびずむ_offvocal':          { sub: '裁剪又贴上的皮，看见开裂的疤' },
  'きゅびびびびずむ':             { sub: '老人正在一边看着你' },
  'きゅびびびびずむ_offvocal':    { sub: '老人正在一边看着你' }
};

// ═══════════════════════════════════════
// 动画角色配置
// ═══════════════════════════════════════
var charConfigs = {
  ame: {
    frameSource: function() { return NGO.jineAmeFrames; },
    path: '/assets/stream/ame/',
    elId: 'ame-avatar-frame',
    loopActions: ['idle_normal', 'idle_iraira', 'idle_anxiety', 'idle_happy'],
    getIdle: function() {
      var stress = NGO.status && typeof NGO.status.stress === 'number' ? NGO.status.stress : 0;
      if (stress >= 80) return 'idle_iraira';
      if (stress >= 60) return 'idle_anxiety';
      if (stress >= 40) return 'idle_normal';
      return 'idle_happy';
    },
    getSuffix: function() {
      var aff = NGO.status && typeof NGO.status.affection === 'number' ? NGO.status.affection : 50;
      if (aff < 20) return '_dark';
      if (aff < 40) return '_low';
      if (aff < 80) return '';
      if (aff < 100) return '_high';
      return '_peak';
    },
    randomActions: ['selfie', 'comic', 'handspinner'],
    randomChance: 0.01,
    dblclickAction: 'smile',
    statTrigger: ['stress', 'affection']
  },

  kangel: {
    frameSource: function() { return NGO.liveKangelFrames; },
    path: '/assets/stream/kangel/',
    elId: 'live-kangel-frame',
    loopActions: ['idle', 'pout', 'start'],
    getIdle: function() { return 'idle'; },
    getSuffix: function() { return ''; },
    randomActions: ['smile', 'pout'],
    randomChance: 0.008,
    dblclickAction: 'smile',
    statTrigger: ['affection'],
    actionMap: function() { return currentTopic && currentTopic.actionMap; }
  },

  dangel: {
    frameSource: function() { return NGO.liveDangelFrames; },
    path: '/assets/stream/dangel/',
    elId: 'live-dangel-frame',
    loopActions: ['idle', 'think', 'start'],
    getIdle: function() { return 'idle'; },
    getSuffix: function() { return ''; },
    randomActions: ['smile'],
    randomChance: 0.008,
    dblclickAction: 'smile',
    statTrigger: ['affection'],
    actionMap: function() { return currentTopic && currentTopic.actionMap; }
  },

  'ame-live': {
    frameSource: function() { return NGO.liveAmeFrames; },
    path: '/assets/stream/ame/',
    elId: 'live-ame-frame',
    loopActions: ['idle', 'idle_normal', 'idle_iraira', 'idle_anxiety', 'idle_happy', 'start'],
    // 直播形态复用私聊 ame 的状态待机素材（同一目录 assets/stream/ame/），
    // 压力选 idle_*、好感选 _dark/_low/_high/_peak 后缀；帧表见 frames-live.js 的合并
    getIdle: function() {
      var stress = NGO.status && typeof NGO.status.stress === 'number' ? NGO.status.stress : 0;
      if (stress >= 80) return 'idle_iraira';
      if (stress >= 60) return 'idle_anxiety';
      if (stress >= 40) return 'idle_normal';
      return 'idle_happy';
    },
    getSuffix: function() {
      var aff = NGO.status && typeof NGO.status.affection === 'number' ? NGO.status.affection : 50;
      if (aff < 20) return '_dark';
      if (aff < 40) return '_low';
      if (aff < 80) return '';
      if (aff < 100) return '_high';
      return '_peak';
    },
    randomActions: ['smile', 'shobon'],
    randomChance: 0.008,
    dblclickAction: 'smile',
    statTrigger: ['stress', 'affection'],
    actionMap: function() { return currentTopic && currentTopic.actionMap; }
  }
};

// ═══════════════════════════════════════
// 聊天 channel 配置
// ═══════════════════════════════════════
var jineChannel = {
  chatAreaSelector: '.jine-chat-area',
  inputSelector: '#jine-input',
  sendBtnSelector: '#jine-send-btn',
  emojiToggleSelector: '#jine-emoji-toggle',
  emojiPopupSelector: '#jine-emoji-popup',
  emojiBtnSelector: '#jine-emoji-popup .jine-emoji-btn',

  msgClass: function(isUser) { return 'jine-msg ' + (isUser ? 'jine-msg-user' : 'jine-msg-ame'); },
  userIcon: function() { return BASE + '/assets/images/icon_bear.png'; },
  botIcon: function() { return BASE + '/assets/images/icon_jine_ame.png'; },
  iconClass: function(isUser) { return isUser ? 'jine-bear-icon' : 'jine-bubble-icon'; },
  bubbleClass: function(isUser) { return 'jine-bubble'; },
  emojiClass: function(isUser) { return 'jine-emoji-img'; },
  timeBarClass: 'jine-date-bar',
  renderTimeBar: function(bar, text) {
    var img = document.createElement('img');
    img.src = BASE + '/assets/images/JINE_date.png';
    img.alt = 'date';
    var span = document.createElement('span');
    span.className = 'jine-date-text';
    span.textContent = text;
    bar.appendChild(img); bar.appendChild(span);
  },

  clearInput: function() {
    var input = document.querySelector('#jine-input');
    if (input) input.value = '';
  },
  getInputValue: function() {
    var input = document.querySelector('#jine-input');
    return input ? input.value.trim() : '';
  },
  getFallbackReply: function() {
    return ameReplies[Math.floor(Math.random() * ameReplies.length)];
  },
  getFallbackEmoji: function() {
    return ameEmojis[Math.floor(Math.random() * ameEmojis.length)];
  },
  playSendAction: function() { playJineAction('talk', { transient: true }); },
  playThinkingAction: function() { ameAnim.setThinking(true); },
  stopThinkingAction: function() { ameAnim.setThinking(false); },
  getMode: function() { return '0'; },
  playSendSfx: function() { playSfx('jine_send_stamp'); },
  playReceiveSfx: function() { playSfx('jine_recieve'); },
  getContextKey: function() { return 'jine'; },
  onAfterReply: function() { if (NGO.getStats) NGO.getStats(); }
};

var liveChannel = {
  chatAreaSelector: '#live-danmaku',
  inputSelector: '#live-chat-input',
  sendBtnSelector: '#live-send-btn',
  emojiToggleSelector: '#live-emoji-toggle',
  emojiPopupSelector: '#live-emoji-popup',
  emojiBtnSelector: '.live-emoji-row .jine-emoji-btn',

  msgClass: function(isUser) { return 'danmaku-item'; },
  userIcon: function() { return BASE + '/assets/images/icon_bear.png'; },
  botIcon: function() { return BASE + '/assets/images/icon_cho.png'; },
  iconClass: function(isUser) { return 'danmaku-tag'; },
  bubbleClass: function(isUser) { return ''; },
  emojiClass: function(isUser) { return 'danmaku-emoji'; },
  timeBarClass: 'danmaku-date-bar',

  onAnswerLifecycle: function(type, text, identity) {
    streamReplyLifecycle(type, text, identity);
  },
  // Standalone non-silent rendering remains compatible; lifecycle sends supply identity.
  onRenderMessage: function(text, isUser, identity) {
    if (!isUser && !identity) showStreamReply(text, streamReplyIdentity);
  },

  clearInput: function() {
    var input = document.querySelector('#live-chat-input');
    if (input) input.value = '';
  },
  getInputValue: function() {
    var input = document.querySelector('#live-chat-input');
    return input ? input.value.trim() : '';
  },
  getFallbackReply: function() {
    return kangelLiveReplies[Math.floor(Math.random() * kangelLiveReplies.length)];
  },
  getFallbackEmoji: function() {
    return kangelLiveEmojis[Math.floor(Math.random() * kangelLiveEmojis.length)];
  },
  playSendAction: function() { playLiveAction('think', { transient: true }); },
  playThinkingAction: function(key) {
    var topic = NGO.topicConfigs.find(function(t) { return t.id === key; });
    var anim = topic && {kangel: kangelAnim, dangel: dangelAnim, ame: ameLiveAnim}[topic.charVariant];
    if (anim) anim.setThinking(true);
  },
  stopThinkingAction: function(key) {
    var topic = NGO.topicConfigs.find(function(t) { return t.id === key; });
    var anim = topic && {kangel: kangelAnim, dangel: dangelAnim, ame: ameLiveAnim}[topic.charVariant];
    if (anim) anim.setThinking(false);
  },
  getMode: function() { return currentTopic && currentTopic.mode ? currentTopic.mode : null; },
  playSendSfx: function() { playSfx('jine_send_stamp'); },
  playReceiveSfx: function() { playSfx('jine_recieve'); },

  // 每个直播主题各自的消息区；思考块跟着所在的行，不再单独缓存
  getContextKey: function() {
    return (typeof currentTopic !== 'undefined' && currentTopic) ? currentTopic.id : null;
  },
  onAfterReply: function() { if (NGO.getStats) NGO.getStats(); }
};
