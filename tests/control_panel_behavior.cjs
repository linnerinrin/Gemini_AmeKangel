/**
 * tests/control_panel_behavior.cjs — P5a 控制面板行为测试
 *
 * 覆盖：
 * 1. 切换「对话模式」写入 NGO.taskMode 并持久化到 localStorage；
 * 2. 「保存」调用正确的接口（/chat/settings/llm）与载荷（active + profiles）；
 * 3. API Key 输入框不被回填明文 key，保存时也不会把掩码当明文提交。
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const root = path.resolve(__dirname, '..');

const PLAINTEXT_KEY = 'sk-p5a-1234567890abcdef-SECRET';
const MASKED_KEY = 'sk-p5a-****cret';

function makeEl(tag) {
  const listeners = {};
  const el = {
    tag: tag || 'div',
    value: '',
    placeholder: '',
    checked: false,
    disabled: false,
    textContent: '',
    className: '',
    style: {},
    classes: new Set(),
    attrs: {},
    id: '',
    addEventListener(name, cb) { (listeners[name] = listeners[name] || []).push(cb); },
    fire(name, event) { (listeners[name] || []).forEach(cb => cb(event || { type: name })); },
    setAttribute(k, v) { el.attrs[k] = v; },
    getAttribute(k) { return el.attrs[k]; },
  };
  el.classList = {
    add(name) { el.classes.add(name); },
    remove(name) { el.classes.delete(name); },
    contains(name) { return el.classes.has(name); },
    toggle(name) { el.classes.has(name) ? el.classes.delete(name) : el.classes.add(name); },
  };
  return el;
}

(async () => {
  // ── fake DOM ──
  const els = {
    'control-window': makeEl('div'),
    'bgm-volume-slider': makeEl('input'),
    'bgm-volume-val': makeEl('span'),
    'bgm-autoplay': makeEl('input'),
    'sfx-volume-slider': makeEl('input'),
    'sfx-volume-val': makeEl('span'),
    'llm-provider': makeEl('select'),
    'llm-model': makeEl('input'),
    'llm-base-url': makeEl('input'),
    'llm-api-key': makeEl('input'),
    'llm-save-btn': makeEl('button'),
    'llm-test-btn': makeEl('button'),
    'llm-status': makeEl('div'),
  };
  els['llm-provider'].value = 'deepseek';
  const radioChat = makeEl('input');
  radioChat.value = 'chat';
  radioChat.checked = true;
  const radioWork = makeEl('input');
  radioWork.value = 'work';

  const storage = new Map();
  const localStorage = {
    getItem: key => (storage.has(key) ? storage.get(key) : null),
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: key => storage.delete(key),
  };

  const registered = {};
  const ctx = vm.createContext({
    console, setTimeout, clearTimeout, JSON, Promise, URL, fetch: undefined,
    document: {
      getElementById: id => els[id] || null,
      querySelectorAll: selector => (selector === 'input[name="task-mode"]' ? [radioChat, radioWork] : []),
      querySelector: () => null,
      addEventListener: () => {},
    },
    localStorage,
    location: { port: '5577', origin: 'http://localhost:5577' },
    showError: msg => { ctx.__lastError = msg; },
  });
  ctx.window = ctx;
  // 与既有行为测试一致：window 就是沙箱自身，NGO 提前挂好再做属性增补，
  // 否则 control.js 里的 `window.NGO = window.NGO || {}` 会覆盖掉测试注入的替身。
  ctx.NGO = {
    WindowManager: { register: (el, cfg) => { registered.control = cfg; }, focus: () => {} },
    setBgmVolume: () => {}, setSfxVolume: () => {}, playSfx: () => {},
  };

  function load(file) {
    vm.runInContext(fs.readFileSync(path.join(root, file), 'utf8'), ctx, { filename: file });
  }

  // ── 记录前端真正发出去的请求 ──
  const requests = [];
  let nextResponse = {};
  ctx.fetch = async (url, options) => {
    const opts = options || {};
    requests.push({ url, method: opts.method || 'GET', body: opts.body ? JSON.parse(opts.body) : null });
    return new Response(JSON.stringify(nextResponse), { status: 200, headers: { 'content-type': 'application/json' } });
  };

  load('static/js/data/api/core.js');
  load('static/js/data/api/settings.js');
  load('static/js/ui/control.js');

  const model = ctx.NGO.controlPanelModel;
  assert.ok(model, 'controlPanelModel 必须导出，供行为测试断言');

  // ═══ 1. 对话模式：写 NGO.taskMode + localStorage ═══
  assert.equal(ctx.NGO.taskMode, 'chat', '默认任务轴是 chat');
  assert.equal(radioChat.checked, true);

  radioWork.checked = true;
  radioWork.fire('change');
  assert.equal(ctx.NGO.taskMode, 'work', '切换模式必须写 NGO.taskMode');
  assert.equal(localStorage.getItem(model.taskModeKey), 'work', '模式必须持久化到 localStorage');

  radioChat.checked = true;
  radioChat.fire('change');
  assert.equal(ctx.NGO.taskMode, 'chat');
  assert.equal(localStorage.getItem(model.taskModeKey), 'chat');

  // 非法值兜底，避免写坏 NGO.taskMode 让 registry 的 getTaskMode 拿到垃圾
  assert.equal(model.normalizeTaskMode('work'), 'work');
  assert.equal(model.normalizeTaskMode('sleep'), 'chat');
  assert.equal(model.normalizeTaskMode(undefined), 'chat');
  assert.equal(model.initialTaskMode(null, null), null, '没有已存值时应返回 null');

  // ═══ 2. 打开面板拉设置：key 输入框不被回填明文 ═══
  const settingsPayload = {
    active: 'deepseek',
    providers: [
      { name: 'deepseek', label: 'DeepSeek' },
      { name: 'openai', label: 'OpenAI' },
      { name: 'custom', label: '自定义（OpenAI 兼容）' },
    ],
    profiles: {
      deepseek: { label: 'DeepSeek', base_url: 'https://api.deepseek.com', model: 'deepseek-v4-flash', model_placeholder: 'deepseek-v4-flash', api_key: MASKED_KEY, configured: true, key_source: 'settings' },
      openai: { label: 'OpenAI', base_url: 'https://api.openai.com/v1', model: 'gpt-4o-mini', model_placeholder: 'gpt-4o-mini', api_key: '', configured: false, key_source: 'none' },
      custom: { label: '自定义（OpenAI 兼容）', base_url: '', model: '', model_placeholder: '', api_key: '', configured: false, key_source: 'none' },
    },
    configured: true,
    key_source: 'settings',
    base_url: 'https://api.deepseek.com',
    model: 'deepseek-v4-flash',
    message: '',
  };
  nextResponse = settingsPayload;
  model.syncLlmSettings();
  await new Promise(resolve => setImmediate(resolve));

  assert.equal(requests.at(-1).url, 'http://localhost:5577/chat/settings/llm');
  assert.equal(requests.at(-1).method, 'GET');
  assert.equal(els['llm-provider'].value, 'deepseek');
  assert.equal(els['llm-model'].value, 'deepseek-v4-flash');
  assert.equal(els['llm-api-key'].value, '', 'API Key 输入框必须为空，不能被回填明文');
  assert.equal(els['llm-api-key'].placeholder.includes(MASKED_KEY), true, '掩码只能作为占位提示');
  assert.equal(els['llm-api-key'].value.includes(PLAINTEXT_KEY), false);
  assert.equal(els['llm-status'].textContent.includes(PLAINTEXT_KEY), false);

  // ═══ 3. 保存：接口 + 载荷（key 未被编辑 => 不提交 api_key）═══
  const savedResponse = Object.assign({}, settingsPayload, { model: 'deepseek-v4-flash' });
  nextResponse = savedResponse;
  els['llm-model'].value = 'deepseek-v4-flash';
  els['llm-save-btn'].fire('click');
  await new Promise(resolve => setImmediate(resolve));
  await new Promise(resolve => setImmediate(resolve));

  const saveRequest = requests.at(-1);
  assert.equal(saveRequest.url, 'http://localhost:5577/chat/settings/llm');
  assert.equal(saveRequest.method, 'POST');
  assert.equal(saveRequest.body.active, 'deepseek');
  assert.deepEqual(Object.keys(saveRequest.body.profiles), ['deepseek']);
  assert.equal(saveRequest.body.profiles.deepseek.model, 'deepseek-v4-flash');
  assert.equal(saveRequest.body.profiles.deepseek.base_url, 'https://api.deepseek.com');
  assert.equal('api_key' in saveRequest.body.profiles.deepseek, false, '没改过输入框就不该提交 api_key');
  assert.equal(JSON.stringify(saveRequest.body).includes(PLAINTEXT_KEY), false);
  assert.equal(JSON.stringify(saveRequest.body).includes('****'), false, '掩码也不能被当成 key 提交');
  assert.match(els['llm-status'].textContent, /已保存/);

  // ═══ 4. 用户真的改了 key：提交明文（后端需要明文才能存）═══
  els['llm-api-key'].value = PLAINTEXT_KEY;
  els['llm-api-key'].fire('input');
  els['llm-save-btn'].fire('click');
  await new Promise(resolve => setImmediate(resolve));
  await new Promise(resolve => setImmediate(resolve));

  assert.equal(requests.at(-1).body.profiles.deepseek.api_key, PLAINTEXT_KEY, '用户改过就必须提交明文');
  // 保存成功后输入框被清空（只留掩码占位），避免明文长时间留在页面上
  assert.equal(els['llm-api-key'].value, '', '保存成功后必须清空 key 输入框');

  // ═══ 5. 纯函数载荷：掩码 key 永不进入请求体 ═══
  const maskedDraft = model.buildSavePayload(settingsPayload, {
    provider: 'deepseek', model: 'm', baseUrl: 'https://api.deepseek.com', apiKey: MASKED_KEY, keyDirty: false,
  });
  assert.equal('api_key' in maskedDraft.profiles.deepseek, false);

  // ═══ 6. 切换厂商：用新厂商的已保存值，key 仍不回填明文 ═══
  els['llm-provider'].value = 'openai';
  els['llm-provider'].fire('change');
  assert.equal(els['llm-model'].value, 'gpt-4o-mini');
  assert.equal(els['llm-base-url'].value, 'https://api.openai.com/v1');
  assert.equal(els['llm-api-key'].value, '');
  assert.equal(model.buildSavePayload(settingsPayload, {
    provider: 'custom', model: 'my-model', baseUrl: 'https://gw.example.com/v1', apiKey: '', keyDirty: false,
  }).profiles.custom.model, 'my-model');

  // ═══ 7. 测试连通性：走 /settings/llm/test，且不把掩码当 key 发 ═══
  nextResponse = { provider: 'openai', model: 'gpt-4o-mini', base_url: 'https://api.openai.com/v1', ok: false, message: 'API Key 无效或已过期（401），请确认后重新填写' };
  els['llm-test-btn'].fire('click');
  await new Promise(resolve => setImmediate(resolve));
  await new Promise(resolve => setImmediate(resolve));

  const testRequest = requests.at(-1);
  assert.equal(testRequest.url, 'http://localhost:5577/chat/settings/llm/test');
  assert.equal(testRequest.method, 'POST');
  assert.equal(testRequest.body.provider, 'openai');
  assert.equal('api_key' in testRequest.body, false, '没改过 key 就不提交 api_key');
  assert.match(els['llm-status'].textContent, /401/);
  // 测试失败也必须走用户可见通道（面板状态文字 + showError），且提示里不能带明文 key
  assert.equal(els['llm-status'].className.includes('error'), true);
  assert.equal(String(ctx.__lastError || '').includes(PLAINTEXT_KEY), false);
  assert.equal(els['llm-status'].textContent.includes(PLAINTEXT_KEY), false);

  // ═══ 8. 回归：面板重获焦点（applySettingsToPanel 重跑）不得抹掉用户刚输入的 key ═══
  // 真实故障：点密码框旁的眼睛图标 / 点保存按钮都会让窗口重新获得焦点 → onFocus →
  // syncLlmSettings → applySettingsToPanel，早期实现无条件 keyInput.value='' + keyDirty=false，
  // 于是"填好 key 再点保存"退化成"没填 key"，后端只能报「尚未配置 API Key」。
  els['llm-provider'].value = 'deepseek';
  els['llm-api-key'].value = PLAINTEXT_KEY;
  els['llm-api-key'].fire('input');
  model.applySettingsToPanel(settingsPayload);   // 不传 clearKey：模拟 focus 触发的重渲染
  assert.equal(els['llm-api-key'].value, PLAINTEXT_KEY, '聚焦重渲染后已输入的 key 必须还在');
  assert.equal(model.buildSavePayload(settingsPayload, {
    provider: 'deepseek', model: 'm', baseUrl: 'https://api.deepseek.com',
    apiKey: els['llm-api-key'].value, keyDirty: true,
  }).profiles.deepseek.api_key, PLAINTEXT_KEY, '重渲染后保存仍须带上明文 key');

  els['llm-save-btn'].fire('click');
  await new Promise(resolve => setImmediate(resolve));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(requests.at(-1).body.profiles.deepseek.api_key, PLAINTEXT_KEY, '重渲染后点保存必须提交明文 key');
  assert.equal(els['llm-api-key'].value, '', '保存成功后仍然要清空输入框');

  console.log('JS control panel behavior: task mode persistence, settings load/save payloads, masked key never echoed, connectivity test, focus re-render keeps typed key passed');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
