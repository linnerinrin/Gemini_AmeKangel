/**
 * ui/control — 控制面板窗口 + 音量滑块绑定 + P5a 模型设置
 */
'use strict';

window.NGO = window.NGO || {};

// ═══════════════════════════════════════
// P5a：对话模式（任务轴）与模型设置
// ═══════════════════════════════════════

/** 对话模式（任务轴）持久化用的 localStorage key */
var TASK_MODE_KEY = 'ngo_task_mode';

/** 把任意值归一化到任务轴合法值；非法值兜底 chat */
function normalizeTaskMode(value) {
  return value === 'work' ? 'work' : 'chat';
}

/** 读 localStorage；读不到返回 null（区别于显式的 'chat'） */
function readTaskMode() {
  try {
    var stored = window.localStorage.getItem(TASK_MODE_KEY);
    return stored === null ? null : normalizeTaskMode(stored);
  } catch (err) {
    return null;
  }
}

function writeTaskMode(mode) {
  try { window.localStorage.setItem(TASK_MODE_KEY, mode); } catch (err) { /* 隐私模式等：忽略 */ }
}

/**
 * 打开面板时的任务轴初值：localStorage 优先，其次沿用已挂在 NGO.taskMode 上的值，
 * 都没有才用 chat。返回 null 表示"没有已存值"，调用方不要覆盖 localStorage。
 */
function initialTaskMode(storedMode, currentMode) {
  if (typeof storedMode === 'string') return normalizeTaskMode(storedMode);
  if (typeof currentMode === 'string') return normalizeTaskMode(currentMode);
  return null;
}

/** 切换对话模式：写 NGO.taskMode（registry.js 的 getTaskMode 读它）+ 持久化 */
function applyTaskMode(mode) {
  var value = normalizeTaskMode(mode);
  window.NGO.taskMode = value;
  writeTaskMode(value);
  return value;
}

var taskModeStored = readTaskMode();
var taskModeInitial = initialTaskMode(taskModeStored, window.NGO.taskMode);
// 即使没有已存值也要落成 'chat'：让 NGO.taskMode 在面板初始化后就永不为 undefined，
// registry.js 的 getTaskMode 兜底逻辑就不会两头都是空的。
applyTaskMode(taskModeInitial !== null ? taskModeInitial : 'chat');

/**
 * 构造保存请求体：只把用户真的改过的厂商字段发出去。
 * - API Key 输入框没被编辑过 => 不带 api_key（后端沿用已保存/.env 的 key），
 *   所以掩码值不会被当成明文回传。
 * - 输入框里是掩码时前端也不提交：只有 keyDirty 为真才带上 api_key。
 */
function buildDraftPayload(settings, provider, form) {
  var stored = ((settings && settings.profiles) || {})[provider] || {};
  var profile = {
    base_url: form.baseUrl || stored.base_url || '',
    model: form.model || stored.model || ''
  };
  if (form.keyDirty) profile.api_key = form.apiKey || '';
  var profiles = {};
  profiles[provider] = profile;
  return { active: provider, profiles: profiles };
}

// ── Window registration ──
NGO.WindowManager.register(document.getElementById('control-window'), {});

// ── BGM volume slider ──
var bgmVolSlider = document.getElementById('bgm-volume-slider');
var bgmVolVal   = document.getElementById('bgm-volume-val');
if (bgmVolSlider && bgmVolVal) {
  bgmVolSlider.addEventListener('input', function() {
    var v = parseInt(bgmVolSlider.value);
    bgmVolVal.textContent = v;
    NGO.setBgmVolume(v / 100);
  });
}

// ── BGM autoplay checkbox ──
var bgmAutoplay = document.getElementById('bgm-autoplay');
if (bgmAutoplay) {
  bgmAutoplay.addEventListener('change', function() {});
}

// ── SFX volume slider ──
var sfxVolSlider = document.getElementById('sfx-volume-slider');
var sfxVolVal = document.getElementById('sfx-volume-val');
if (sfxVolSlider && sfxVolVal) {
  sfxVolSlider.addEventListener('input', function() {
    var value = parseInt(sfxVolSlider.value, 10);
    sfxVolVal.textContent = value;
    NGO.setSfxVolume(value / 100);
  });
}

// ═══════════════════════════════════════
// P5a：模型设置面板
// ═══════════════════════════════════════

// 上一次 GET 回来的设置（key 是掩码）；保存/测试前都以它为准
var llmSettingsCache = null;
// API Key 输入框是否被用户改过；没改过就不提交该字段，避免把掩码当明文存下去
var llmKeyDirty = false;

function controlEl(id) { return document.getElementById(id); }

function setStatus(text, kind) {
  var el = controlEl('llm-status');
  if (!el) return;
  el.textContent = text || '';
  el.className = 'control-status' + (kind ? ' ' + kind : '');
}

/**
 * 把服务端返回的设置渲染到面板（key 只显示掩码，绝不显示明文）。
 *
 * opts.clearKey=true 时才会清空 key 输入框（保存成功后用：key 已经存到服务端了）。
 * 默认**不清空已输入的 key**：本函数会在面板每次获得焦点时被 syncLlmSettings 调用，
 * 早期实现无条件清空 + 复位 keyDirty，导致"填好 key → 点眼睛图标/点保存 → key 被抹掉"
 * → 保存请求不带 api_key → 后端只能报"尚未配置 API Key"。
 */
function applySettingsToPanel(settings, opts) {
  llmSettingsCache = settings || null;
  var select = controlEl('llm-provider');
  var modelInput = controlEl('llm-model');
  var baseInput = controlEl('llm-base-url');
  var keyInput = controlEl('llm-api-key');
  if (!settings) return;

  if (select) select.value = settings.active;
  var profile = (settings.profiles || {})[settings.active] || {};
  if (modelInput) modelInput.value = profile.model || '';
  if (baseInput) baseInput.value = profile.base_url || '';
  if (keyInput) {
    // 用户已经输入过新 key（dirty）就保留原样，否则清空并复位
    if ((opts && opts.clearKey) || !llmKeyDirty) {
      keyInput.value = '';
      llmKeyDirty = false;
    }
    // 有掩码就显示掩码当占位提示
    keyInput.placeholder = profile.api_key ? ('已保存：' + profile.api_key + '（留空沿用）') : '请粘贴 API Key';
  }
  setStatus(settings.message || '', settings.configured ? '' : 'error');
}

/**
 * 把面板上的当前值整理成保存请求体（纯函数，供行为测试直接调用）。
 * 掩码 key 永远不会进入请求体：只有 keyDirty 为真才带上 api_key。
 */
function buildSavePayload(settings, form) {
  var active = form.provider || (settings && settings.active) || 'deepseek';
  return buildDraftPayload(settings, active, form);
}

// 供行为测试调用的纯逻辑出口（不依赖 DOM）
NGO.controlPanelModel = {
  taskModeKey: TASK_MODE_KEY,
  normalizeTaskMode: normalizeTaskMode,
  initialTaskMode: initialTaskMode,
  applyTaskMode: applyTaskMode,
  buildSavePayload: buildSavePayload,
  buildDraftPayload: buildDraftPayload,
  // DOM 相关的两个函数也导出：行为测试用它验证"key 输入框不会被回填明文"
  applySettingsToPanel: applySettingsToPanel,
  syncLlmSettings: syncLlmSettings
};

function syncLlmSettings() {
  if (!NGO.settingsAPI || typeof NGO.settingsAPI.getSettings !== 'function') {
    setStatus('设置接口未加载（settings.js）', 'error');
    return;
  }
  setStatus('读取中…');
  NGO.settingsAPI.getSettings().then(function(settings) {
    applySettingsToPanel(settings);
  }).catch(function(err) {
    setStatus(err && err.message ? err.message : '读取模型设置失败', 'error');
  });
}

(function bindLlmPanel() {
  var select = controlEl('llm-provider');
  var modelInput = controlEl('llm-model');
  var baseInput = controlEl('llm-base-url');
  var keyInput = controlEl('llm-api-key');
  if (!select || !modelInput || !baseInput || !keyInput) return;

  // 切换厂商：把该厂商已保存的 base_url/model 填进来，key 留空等用户决定
  select.addEventListener('change', function() {
    var profile = ((llmSettingsCache && llmSettingsCache.profiles) || {})[select.value] || {};
    modelInput.value = profile.model || '';
    baseInput.value = profile.base_url || '';
    keyInput.value = '';
    keyInput.placeholder = profile.api_key ? ('已保存：' + profile.api_key + '（留空沿用）') : '请粘贴 API Key';
    llmKeyDirty = false;
    setStatus('');
  });

  keyInput.addEventListener('input', function() { llmKeyDirty = true; });

  var saveBtn = controlEl('llm-save-btn');
  if (saveBtn) {
    saveBtn.addEventListener('click', function() {
      if (!NGO.settingsAPI) return;
      var payload = buildSavePayload(llmSettingsCache, {
        provider: select.value,
        model: modelInput.value.trim(),
        baseUrl: baseInput.value.trim(),
        apiKey: keyInput.value,
        keyDirty: llmKeyDirty
      });
      setStatus('保存中…');
      saveBtn.disabled = true;
      NGO.settingsAPI.saveSettings(payload.active, payload.profiles).then(function(settings) {
        // 保存成功后 key 已在服务端，这里才应该清空输入框
        applySettingsToPanel(settings, {clearKey: true});
        setStatus('已保存：当前使用 ' + settings.active + ' / ' + settings.model, 'ok');
      }).catch(function(err) {
        setStatus(err && err.message ? err.message : '保存失败', 'error');
        if (NGO.showError) NGO.showError(err && err.message ? err.message : '保存失败');
      }).then(function() { saveBtn.disabled = false; });
    });
  }

  var testBtn = controlEl('llm-test-btn');
  if (testBtn) {
    testBtn.addEventListener('click', function() {
      if (!NGO.settingsAPI) return;
      var active = select.value;
      var stored = ((llmSettingsCache && llmSettingsCache.profiles) || {})[active] || {};
      var candidate = {
        base_url: baseInput.value.trim() || stored.base_url || '',
        model: modelInput.value.trim() || stored.model || ''
      };
      setStatus('测试中…');
      testBtn.disabled = true;
      NGO.settingsAPI.testSettings(
        active,
        candidate.base_url,
        candidate.model,
        llmKeyDirty ? keyInput.value : ''
      ).then(function(result) {
        setStatus(result.message || (result.ok ? '连接成功' : '测试失败'), result.ok ? 'ok' : 'error');
        if (!result.ok && NGO.showError) NGO.showError(result.message || '测试失败');
      }).catch(function(err) {
        setStatus(err && err.message ? err.message : '测试失败', 'error');
        if (NGO.showError) NGO.showError(err && err.message ? err.message : '测试失败');
      }).then(function() { testBtn.disabled = false; });
    });
  }
})();

// 对话模式单选：切换即写 NGO.taskMode（下一次发送就按新任务轴走）
(function bindTaskModeRadios() {
  var radios = document.querySelectorAll('input[name="task-mode"]');
  if (!radios || !radios.length) return;
  var current = normalizeTaskMode(NGO.taskMode);
  radios.forEach(function(radio) {
    radio.checked = radio.value === current;
    radio.addEventListener('change', function() {
      if (!radio.checked) return;
      applyTaskMode(radio.value);
    });
  });
})();

// 打开控制面板时拉一次设置：以后端文件为准（刷新后仍然保留）
(function registerPanelFocus() {
  var win = document.getElementById('control-window');
  if (!win) return;
  var cfg = {
    onFocus: function() {
      // 用户正在填 key 时不要重拉：既没必要，又会在渲染时干扰输入框
      if (llmKeyDirty) return;
      syncLlmSettings();
    }
  };
  if (NGO.WindowManager.register) NGO.WindowManager.register(win, cfg);
})();

// ── Global click SFX ──
document.addEventListener('click', function() {
  NGO.playSfx('Click');
});
