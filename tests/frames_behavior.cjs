/**
 * frames_behavior.cjs — 帧表守卫
 *
 * 守住三件事，防止"状态 -> 帧动画"设计静默失效：
 *  1. 帧表引用的 PNG 全部存在（图片被改名/删除会立刻报错）
 *  2. ame 私聊的 4×5 状态矩阵（压力 4 档 idle 家族 × 好感 5 档变体）每格都能取到帧
 *  3. 各角色声明的动作键（loopActions / randomActions / dblclickAction / getIdle 返回值 /
 *     直播主题 actionMap）在其帧表里存在 —— 这条正是为了抓住 liveAmeFrames 曾缺 smile 的缺口
 *  4. 帧表缺动作导致退化为 idle 时要 console.warn（后缀回退与 start 探测不打扰）
 */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');

function newCtx(extra) {
  const ctx = { window: null, NGO: {}, console: { log() {}, warn() {} }, BASE: '' };
  Object.assign(ctx, extra || {});
  ctx.window = ctx;
  vm.createContext(ctx);
  return ctx;
}

function run(ctx, rel) {
  vm.runInContext(fs.readFileSync(path.join(ROOT, rel), 'utf8'), ctx, { filename: rel });
  return ctx;
}

const ctx = newCtx();
['static/js/data/frames-jine.js', 'static/js/data/frames-live.js',
 'static/js/data/topics.js', 'static/js/data/registry.js'].forEach(rel => run(ctx, rel));
const NGO = ctx.NGO;
const charConfigs = ctx.charConfigs;

// ── 1. 帧表引用的 PNG 必须存在 ──
const TABLES = [
  ['jineAmeFrames', NGO.jineAmeFrames, 'static/assets/stream/ame'],
  ['liveAmeFrames', NGO.liveAmeFrames, 'static/assets/stream/ame'],
  ['liveKangelFrames', NGO.liveKangelFrames, 'static/assets/stream/kangel'],
  ['liveDangelFrames', NGO.liveDangelFrames, 'static/assets/stream/dangel'],
];
const missingFiles = [];
for (const [name, table, dir] of TABLES) {
  assert.ok(table, `帧表 ${name} 不存在`);
  for (const [key, files] of Object.entries(table)) {
    assert.ok(Array.isArray(files) && files.length, `${name}.${key} 不是非空帧数组`);
    for (const file of files) {
      if (!fs.existsSync(path.join(ROOT, dir, file))) missingFiles.push(`${name}.${key} -> ${file}`);
    }
  }
}
assert.equal(missingFiles.length, 0, '帧表引用的 PNG 缺失:\n' + missingFiles.join('\n'));

// ── 2. ame 私聊 4×5 状态矩阵 ──
const FAMILIES = ['idle_happy', 'idle_normal', 'idle_anxiety', 'idle_iraira'];
const SUFFIXES = ['_dark', '_low', '', '_high', '_peak'];
const missingCells = [];
for (const family of FAMILIES) {
  for (const suffix of SUFFIXES) {
    if (!NGO.jineAmeFrames[family + suffix]) missingCells.push(family + suffix);
  }
}
assert.equal(missingCells.length, 0, '状态矩阵缺格:\n' + missingCells.join('\n'));

// 阈值 -> 实际取到的键，必须落在矩阵内（含边界）
const ame = charConfigs.ame;
const observed = new Set();
for (const stress of [0, 39, 40, 59, 60, 79, 80, 100]) {
  for (const aff of [0, 19, 20, 39, 40, 79, 80, 99, 100]) {
    NGO.status = { stress, affection: aff };
    const key = ame.getIdle() + ame.getSuffix();
    observed.add(key);
    assert.ok(NGO.jineAmeFrames[key], `stress=${stress} affection=${aff} 取到不存在的键 ${key}`);
  }
}
assert.equal(observed.size, 20, `状态矩阵应覆盖 20 格，实际 ${observed.size}: ${[...observed].join(',')}`);

// ── 3. 声明的动作键必须存在 ──
const CHAR_TABLES = {
  ame: ['jineAmeFrames', NGO.jineAmeFrames],
  kangel: ['liveKangelFrames', NGO.liveKangelFrames],
  dangel: ['liveDangelFrames', NGO.liveDangelFrames],
  'ame-live': ['liveAmeFrames', NGO.liveAmeFrames],
};
const missingActions = [];
for (const [charName, [tableName, table]] of Object.entries(CHAR_TABLES)) {
  const cfg = charConfigs[charName];
  assert.ok(cfg, `charConfigs.${charName} 不存在`);
  const declared = new Set();
  (cfg.loopActions || []).forEach(a => declared.add(a));
  (cfg.randomActions || []).forEach(a => declared.add(a));
  if (cfg.dblclickAction) declared.add(cfg.dblclickAction);
  // setThinking 用的动作：私聊 ame 播 talk，直播角色播 think
  declared.add(charName === 'ame' ? 'talk' : 'think');
  // getIdle 的所有可能返回值
  for (const stress of [0, 40, 60, 80, 100]) {
    NGO.status = { stress, affection: 50 };
    declared.add(cfg.getIdle());
  }
  for (const action of declared) {
    if (!table[action]) missingActions.push(`${charName}: 动作 "${action}" 不在 ${tableName}`);
  }
}

// 直播主题的 actionMap 值必须存在（charVariant -> 动画实例 -> 帧表）
const VARIANT_TABLES = {
  kangel: ['liveKangelFrames', NGO.liveKangelFrames],
  dangel: ['liveDangelFrames', NGO.liveDangelFrames],
  ame: ['liveAmeFrames', NGO.liveAmeFrames],
};
for (const topic of NGO.topicConfigs) {
  if (!topic.charVariant) continue;
  const entry = VARIANT_TABLES[topic.charVariant];
  assert.ok(entry, `topic ${topic.id} 的 charVariant "${topic.charVariant}" 没有对应帧表`);
  const [tableName, table] = entry;
  for (const [key, action] of Object.entries(topic.actionMap || {})) {
    if (!table[action]) {
      missingActions.push(`topic ${topic.id}(${topic.title}) actionMap.${key}="${action}" 不在 ${tableName}`);
    }
  }
}
assert.equal(missingActions.length, 0, '声明了但帧表里没有的动作:\n' + missingActions.join('\n'));

// ── 4. 退化到 idle 必须可见 ──
const warnings = [];
const probeEl = { src: '', closest: () => null };
const probeTable = { idle: ['idle.png'], talk: ['talk.png'], smile_dark: ['smile_dark.png'] };
const probeCfg = {
  frameSource: () => probeTable, path: '/', elId: 'probe', loopActions: ['idle', 'talk'],
  randomActions: [], randomChance: 0, dblclickAction: null,
  getIdle: () => 'idle', getSuffix: () => '',
};
const actx = newCtx({
  NGO: { _statHandlers: [] },
  charConfigs: { ame: probeCfg, kangel: probeCfg, dangel: probeCfg, 'ame-live': probeCfg },
  document: { getElementById: () => probeEl },
  setInterval: () => 1, clearInterval: () => {}, setTimeout: () => 1, clearTimeout: () => {},
  console: { log() {}, warn: (msg) => warnings.push(msg) },
});
run(actx, 'static/js/service/animation.js');
const anim = actx.createCharAnim(probeCfg);

anim.playAction('nonexistent');            // 基名缺失 -> 退化 idle，必须提示
assert.equal(warnings.length, 1, '帧表缺动作时应恰好 warn 一次，实际: ' + JSON.stringify(warnings));
assert.match(warnings[0], /nonexistent/);

probeCfg.getSuffix = () => '_dark';
anim.playAction('talk');                   // talk_dark 缺失但 talk 存在 -> 设计内回退，不提示
assert.equal(warnings.length, 1, '后缀回退不应提示: ' + JSON.stringify(warnings));
probeCfg.getSuffix = () => '';

anim.start();                              // start 是可选探测动作，缺失不提示
assert.equal(warnings.length, 1, 'start 探测不应提示: ' + JSON.stringify(warnings));

console.log('frames behavior passed: PNG 完整性, 4x5 状态矩阵, 声明动作存在性, 退化提示');
