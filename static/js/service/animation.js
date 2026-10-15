/**
 * services/animation-core — 共享帧动画引擎
 * 通过 charConfig 对象区分 ame / kangel / dangel 的帧表和路径
 */
'use strict';

// ── 核心帧播放 ──
function playFrames(config) {
  if (!config.frames || !config.frames.length || !config.el) return;
  if (config._state.timer) { clearInterval(config._state.timer); config._state.timer = null; }

  var fps = config.fps || 300;
  var i = 0;
  config.el.src = BASE + config.path + config.frames[0];
  if (config.onFrame) config.onFrame(i, config.frames.length);

  config._state.timer = setInterval(function() {
    i++;
    if (i >= config.frames.length) {
      if (config.oneshot) {
        clearInterval(config._state.timer);
        config._state.timer = null;
        // 单次动画播完后回到循环动作，避免一直卡在这个动画上
        if (config.onFinish) config.onFinish();
        return;
      }
      i = 0;
      if (config.onLoopEnd) {
        var next = config.onLoopEnd();
        if (next) { playFrames(config._build(next)); return; }
      }
    }
    config.el.src = BASE + config.path + config.frames[i];
    if (config.onFrame) config.onFrame(i, config.frames.length);
  }, fps);
}

// ── 创建角色动画实例（charConfigs 在 registry.js 中定义）──
function createCharAnim(cfg) {
  var state = { timer: null, startTimer: null, generation: 0, thinking: false };

  function stop() {
    state.thinking = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    if (state.startTimer) { clearTimeout(state.startTimer); state.startTimer = null; }
    state.generation++;
  }

  function _build(action, opts) {
    opts = opts || {};
    var framesObj = cfg.frameSource();
    var actionName = action;
    var am = cfg.actionMap ? cfg.actionMap() : null;

    if (am && am[action] !== undefined) actionName = am[action];

    var suffix = cfg.getSuffix();
    var key = suffix ? actionName + suffix : actionName;
    var frames = framesObj[key] || framesObj[actionName];

    var idleAction = (am && am.idle) || 'idle';
    if (!frames) {
      var fallback = cfg.getIdle ? cfg.getIdle() : idleAction;
      frames = framesObj[fallback + suffix] || framesObj[fallback] || framesObj[idleAction];
      // 后缀回退（_dark/_high 等只画了 idle/talk）属于设计内行为；只有连基名都取不到、
      // 动作被静默降级成 idle 时才提示，便于发现帧表漏登记（liveAmeFrames 曾缺 smile）。
      // start 是 start() 的可选探测动作，用 optional 标记豁免。
      if (!framesObj[actionName] && !opts.optional &&
          typeof console !== 'undefined' && console.warn) {
        console.warn('[animation] 帧表缺少动作 "' + actionName + '"，已退化为 ' + fallback +
                     '（' + cfg.elId + '）');
      }
    }

    // 循环动作(idle 等)持续播放；其余动作(点击/发送/情绪反应)只播一轮就回到 idle
    var oneshot = !opts.loop && (cfg.loopActions.indexOf(actionName) === -1 || !!opts.transient);

    return {
      frames: frames,
      path: cfg.path,
      el: document.getElementById(cfg.elId),
      fps: 300,
      oneshot: oneshot,
      onFinish: oneshot ? function() {
        playAction(cfg.getIdle ? cfg.getIdle() : idleAction);
      } : null,
      onLoopEnd: function() {
        if (opts.loop) return null;
        if (cfg.randomActions && cfg.randomActions.length && Math.random() < cfg.randomChance) {
          var pick = cfg.randomActions[Math.floor(Math.random() * cfg.randomActions.length)];
          return pick;
        }
        return cfg.getIdle ? cfg.getIdle() : 'idle';
      },
      _state: state,
      _build: _build
    };
  }

  function playAction(action, opts) {
    opts = opts || {};
    if (state.thinking && !opts.loop) return;
    if (state.startTimer) {
      clearTimeout(state.startTimer);
      state.startTimer = null;
      state.generation++;
    }
    if (action === 'idle' && cfg.getIdle) action = cfg.getIdle();
    var fc = _build(action, opts);
    if (!fc.frames) return;
    playFrames(fc);
  }

  function setThinking(active) {
    state.thinking = !!active;
    if (state.thinking) {
      playAction(cfg.elId === 'ame-avatar-frame' ? 'talk' : 'think', { loop: true });
    } else {
      playAction('idle');
    }
  }

  function start() {
    stop();
    var generation = state.generation;
    var idleAction = cfg.getIdle ? cfg.getIdle() : 'idle';
    var fc = _build('start', { optional: true });
    if (!fc.frames) { playAction(idleAction); return; }
    fc.oneshot = true;
    fc.onFinish = null;   // 开场动画由下面的 startTimer 负责回到 idle
    var duration = fc.frames.length * (fc.fps || 300);
    playFrames(fc);
    state.startTimer = setTimeout(function() {
      state.startTimer = null;
      if (generation !== state.generation) return;
      playAction(idleAction);
    }, duration + 50);
  }

  // dblclick
  if (cfg.dblclickAction) {
    var bodyEl = document.getElementById(cfg.elId);
    if (bodyEl) {
      var winBody = bodyEl.closest('.window-body');
      if (winBody) {
        // 用闭包保存
        (function(action) {
          winBody.addEventListener('dblclick', function() {
            if (!saveSelected) return;
            playAction(action);
          });
        })(cfg.dblclickAction);
      }
    }
  }

  // stat handlers
  if (cfg.statTrigger && cfg.statTrigger.length) {
    NGO._statHandlers.push(function(name) {
      if (cfg.statTrigger.indexOf(name) !== -1) playAction('idle');
    });
  }

  return { playAction: playAction, setThinking: setThinking, start: start, stop: stop };
}

// ── 全局入口（兼容旧函数签名） ──
var ameAnim = createCharAnim(charConfigs.ame);
var kangelAnim = createCharAnim(charConfigs.kangel);
var dangelAnim = createCharAnim(charConfigs.dangel);
var ameLiveAnim = createCharAnim(charConfigs['ame-live']);

function startAmeAnimation() {
  ameAnim.start();
}

function stopAmeAnimation() {
  ameAnim.stop();
}

function stopLiveAnimation() {
  kangelAnim.stop();
  dangelAnim.stop();
  ameLiveAnim.stop();
}

function startLiveAnimation() {
  if (!currentTopic) return;
  var char = currentTopic.charVariant;
  ['kangel', 'dangel', 'ame'].forEach(function(name) {
    var el = document.getElementById('live-' + name + '-frame');
    if (!el) return;
    if (name === char) el.classList.remove('hidden');
    else el.classList.add('hidden');
  });
  var anim = { kangel: kangelAnim, dangel: dangelAnim, ame: ameLiveAnim }[char];
  if (anim) anim.start();
}

// 兼容旧代码中的 playJineAction / playLiveAction 全局调用
function playJineAction(action, opts) {
  ameAnim.playAction(action, opts);
}

function playLiveAction(action, opts) {
  if (!currentTopic) return;
  var anim = { kangel: kangelAnim, dangel: dangelAnim, ame: ameLiveAnim }[currentTopic.charVariant];
  if (anim) anim.playAction(action, opts);
}

// ── Live 特有的 affection 变化反应 ──
NGO._statHandlers.push(function(name, oldVal, newVal) {
  if (name !== 'affection' || !currentTopic) return;
  var anim = { kangel: kangelAnim, dangel: dangelAnim, ame: ameLiveAnim }[currentTopic.charVariant];
  if (!anim) return;
  if (newVal > oldVal) anim.playAction('smile');
  else if (newVal < oldVal) anim.playAction('shobon');
});
