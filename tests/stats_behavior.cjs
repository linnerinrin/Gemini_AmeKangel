/**
 * stats_behavior.cjs — 状态值 -> 角色表现 的边界
 *
 * 重点守住修复过的一个 bug：`(NGO.status.affection) || 50` 会把合法的 0 当成缺失值，
 * 导致好感为 0 时永远拿不到 `_dark`。这里同时钉住阈值边界与缺省回退。
 */
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');

const ROOT = path.resolve(__dirname, '..');
const ctx = { window: null, NGO: {}, console: { log() {}, warn() {} }, BASE: '' };
ctx.window = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(ROOT, 'static/js/data/registry.js'), 'utf8'), ctx,
                { filename: 'registry.js' });

const NGO = ctx.NGO;
const ame = ctx.charConfigs.ame;

// 合法的 0 必须被当作数值（修复前 stress=0 -> 20、affection=0 -> 50）
NGO.status = { followers: 0, stress: 0, affection: 0, mental: 0 };
assert.equal(ame.getIdle(), 'idle_happy', 'stress=0 应是 idle_happy');
assert.equal(ame.getSuffix(), '_dark', 'affection=0 应是 _dark');

// 好感阈值（含边界）
for (const [aff, expected] of [
  [0, '_dark'], [19, '_dark'], [20, '_low'], [39, '_low'], [40, ''],
  [79, ''], [80, '_high'], [99, '_high'], [100, '_peak'], [140, '_peak'],
]) {
  NGO.status = { stress: 50, affection: aff };
  assert.equal(ame.getSuffix(), expected, `affection=${aff} 的后缀`);
}

// 压力阈值（含边界）
for (const [stress, expected] of [
  [0, 'idle_happy'], [39, 'idle_happy'], [40, 'idle_normal'], [59, 'idle_normal'],
  [60, 'idle_anxiety'], [79, 'idle_anxiety'], [80, 'idle_iraira'], [100, 'idle_iraira'],
]) {
  NGO.status = { stress: stress, affection: 50 };
  assert.equal(ame.getIdle(), expected, `stress=${stress} 的 idle`);
}

// status 缺失/为 null 时回退到默认值 0 / 50，而不是抛错
delete NGO.status;
assert.equal(ame.getIdle(), 'idle_happy');
assert.equal(ame.getSuffix(), '');
NGO.status = null;
assert.equal(ame.getIdle(), 'idle_happy');
assert.equal(ame.getSuffix(), '');

// 直播角色不参与状态映射
for (const name of ['kangel', 'dangel', 'ame-live']) {
  const cfg = ctx.charConfigs[name];
  NGO.status = { stress: 100, affection: 0 };
  assert.equal(cfg.getIdle(), 'idle', `${name} 的 idle 应恒为 idle`);
  assert.equal(cfg.getSuffix(), '', `${name} 的后缀应恒为空`);
}

console.log('stats behavior passed: 0 值兜底、好感/压力阈值边界、缺省回退与直播隔离');
