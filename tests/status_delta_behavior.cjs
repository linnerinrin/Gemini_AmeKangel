const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');

function element(className) {
  const listeners = {};
  const el = { className, children: [], attrs: {}, style: {}, textContent: '', offsetWidth: 1, restarts: 0,
    appendChild(child) { this.children.push(child); child.parent = this; },
    querySelector(selector) {
      const wanted = selector.replace(/^\./, '');
      return this.children.find(child => child.className.split(/\s+/).indexOf(wanted) !== -1) || null;
    },
    setAttribute(key, val) { this.attrs[key] = val; },
    addEventListener(name, cb) { listeners[name] = cb; },
    fire(name) { if (listeners[name]) listeners[name](); },
  };
  const classes = new Set(className.split(/\s+/).filter(Boolean));
  el.classList = {
    add(name) { if (name === 'run') el.restarts++; classes.add(name); },
    remove(name) { classes.delete(name); },
    contains(name) { return classes.has(name); },
  };
  el.closest = selector => (selector === '.stats-val-area' && el.parent &&
    el.parent.className.split(/\s+/).indexOf('stats-val-area') !== -1) ? el.parent : null;
  return el;
}

// 两个面板：任务管理器与直播面板各一套 stats-val-area / stats-label / stats-core
const names = ['followers', 'affection', 'stress', 'mental'];
const cores = {}, fills = {}, areas = {}, labels = {};
for (const name of names) {
  cores[name] = [element('stats-core'), element('stats-core')];
  fills[name] = [element('stats-bar-fill'), element('stats-bar-fill')];
  areas[name] = [element('stats-val-area'), element('stats-val-area')];
  labels[name] = [element('stats-label'), element('stats-label')];
  cores[name].forEach((core, i) => {
    core.parent = areas[name][i];
    areas[name][i].appendChild(labels[name][i]);
  });
}

let noDom = false;
const ctx = { NGO: {}, saveSelected: true, Number, console, document: {
  querySelectorAll(selector) {
    if (noDom) return [];
    const match = /\[data-stat="(.*?)"\]/.exec(selector);
    if (!match) return [];
    return selector.startsWith('.stats-core') ? cores[match[1]] : fills[match[1]];
  },
  createElement() { return element(''); },
}};
ctx.window = ctx;
vm.createContext(ctx);
for (const file of ['static/js/service/stats.js', 'static/js/service/status-delta.js']) {
  vm.runInContext(fs.readFileSync(path.join(root, file), 'utf8'), ctx, { filename: file });
}

function delta(name, index = 0) { return labels[name][index].querySelector('.stats-delta'); }
function check(name, from, to, text, tone) {
  ctx.NGO.setStat(name, from, { silent: true });
  ctx.NGO.setStat(name, to);
  for (const index of [0, 1]) {
    const el = delta(name, index);
    assert.ok(el, `missing ${name} in panel ${index}`);
    assert.equal(el.textContent, text);
    assert.equal(el.attrs['data-tone'], tone);
    assert.equal(el.parent, labels[name][index], '浮字必须挂在 <stats-label> 内，才紧贴标签文字');
  }
}
check('followers', 0, 1234, '+1,234', 'good');
check('followers', 1234, 1230, '-4', 'bad');
check('affection', 50, 55, '+5', 'good');
check('affection', 55, 52, '-3', 'bad');
check('stress', 10, 5, '-5', 'good');
check('stress', 5, 8, '+3', 'bad');
check('mental', 10, 8, '-2', 'good');
check('mental', 8, 9, '+1', 'bad');
assert.equal(cores.followers[0].textContent, '1,230');
assert.equal(fills.mental[1].style.height, '9%');

const el = delta('affection');
const count = el.restarts;
ctx.NGO.setStat('affection', 53);
assert.equal(labels.affection[0].children.length, 1, '每个标签只保留一个浮字元素');
assert.equal(delta('affection'), el, '复用同一个元素');
assert.equal(el.restarts, count + 1, '再次触发应重启动画');
assert.equal(el.textContent, '+1');
ctx.NGO.setStat('affection', 53);
assert.equal(el.restarts, count + 1, '增量为 0 不重启动画');
ctx.NGO.setStat('affection', 80, { silent: true });
assert.equal(ctx.NGO.status.affection, 80, '静默刷新仍写入数值');
assert.equal(cores.affection[0].textContent, 80, '静默刷新仍更新显示');
assert.equal(el.restarts, count + 1, '静默刷新不播动画');
el.fire('animationend');
assert.equal(el.classList.contains('run'), false);

// 回归守卫 1：浮字必须挂在 <stats-label> 上。曾经用 left:100% 定位到 .stats-val-area 右缘，
// 那里是进度条列，被不透明的 .stats-bar-bg（定位元素、DOM 靠后）整块盖住——动画在播却看不见。
const js = fs.readFileSync(path.join(root, 'static/js/service/status-delta.js'), 'utf8');
assert.match(js, /querySelector\('\.stats-label'\)/, '浮字必须锚定 <stats-label>');

// 回归守卫 2：CSS 不能用会落到进度条上的定位方式，并保留 z-index 兜底。
const css = fs.readFileSync(path.join(root, 'static/css/desktop.css'), 'utf8')
  .replace(/\/\*[\s\S]*?\*\//g, '');   // 去注释，避免注释里的字样影响断言
const deltaRule = /\.stats-delta\s*\{([^}]*)\}/.exec(css);
assert.ok(deltaRule, 'desktop.css 缺少 .stats-delta 规则');
assert.match(deltaRule[1], /position:\s*relative/, '浮字应随标签文字留在文档流内');
assert.doesNotMatch(deltaRule[1], /left:\s*100%/, '不得用 left:100%（会落到进度条上被盖住）');
assert.match(deltaRule[1], /z-index:\s*\d+/, '.stats-delta 需要 z-index 兜底遮挡');
assert.match(css, /@keyframes\s+stats-delta-float\s*\{/, '缺少 stats-delta-float 关键帧');

let observed = null;
const unsubscribe = ctx.NGO.onStatusChange((name, oldVal, value) => { observed = [name, oldVal, value]; });
noDom = true;
ctx.NGO.setStat('affection', 81);
assert.equal(observed[0], 'affection', '缺少数值 DOM 时也不能吞掉钩子');
assert.equal(observed[1], 80);
assert.equal(observed[2], 81);
unsubscribe();
observed = null;
ctx.NGO.setStat('affection', 82);
assert.equal(observed, null, '取消订阅后不再回调');

(async () => {
  noDom = false;
  ctx.NGO.get_status = { getStatus: () => Promise.resolve({ followers: 10, affection: 90, stress: 2, mental: 7 }) };
  const before = delta('followers').restarts;
  ctx.NGO.getStats({ silent: true });
  await Promise.resolve();
  assert.equal(ctx.NGO.status.followers, 10);
  assert.equal(delta('followers').restarts, before);
  ctx.NGO.get_status = { getStatus: () => Promise.resolve({ followers: 12 }) };
  ctx.NGO.getStats();
  await Promise.resolve();
  assert.equal(delta('followers').textContent, '+2');
  assert.equal(delta('followers').restarts, before + 1);
  console.log('Status delta behavior passed: 贴标签定位, 双面板, 配色, 重启, 静默刷新, 无 DOM 仍触发');
})().catch(err => { console.error(err); process.exitCode = 1; });
