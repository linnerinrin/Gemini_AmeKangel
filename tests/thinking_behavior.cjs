const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {El, timers} = require('./stream_fake_dom.cjs');

const clock = timers();
let selection = {isCollapsed: true};
const ctx = vm.createContext({
  console,
  setTimeout: clock.setTimeout,
  clearTimeout: clock.clearTimeout,
  document: {getElementById: () => null, createElement: t => new El(t), createTextNode: t => { const el = new El('#text'); el.textContent = t; return el; }},
  getSelection: () => selection,
});
ctx.window = ctx;
const load = f => vm.runInContext(fs.readFileSync(path.join(__dirname, '..', f), 'utf8'), ctx);
load('static/js/service/text.js');
load('static/js/service/thinking.js');
const NGO = ctx.NGO;

// ── 渲染器本身：同尾项不算前缀，替换/原地编辑/清空都要正确 ──
const panel = new El();
const text = t => ({kind: 'text', text: t});
NGO.renderThinking(panel, [text('A'), text('END')]);
NGO.renderThinking(panel, [text('B'), text('END')]);
assert.equal(panel.textContent, 'BEND');
const ordinary = [text('before'), text('END')];
NGO.renderThinking(panel, ordinary);
ordinary[0].text = 'after';
NGO.renderThinking(panel, ordinary);
assert.equal(panel.textContent, 'afterEND');
NGO.renderThinking(panel, [text('short')]);
assert.equal(panel.textContent, 'short');
NGO.renderThinking(panel, []);
assert.equal(panel.textContent, '');

// 带 identity 的同一数组只追加，已有节点必须复用
const idA = {}, idB = {};
const a = [{kind: 'node', name: 'agent'}, text('A')];
NGO.renderThinking(panel, a, idA);
const header = panel.children[0], group = panel.children[1], first = group.children[0];
a.push(text('END'));
NGO.renderThinking(panel, a, idA);
assert.equal(panel.children[0], header);
assert.equal(group.children[0], first);
NGO.renderThinking(panel, [{kind: 'node', name: 'agent'}, text('B'), text('END')], idB);
assert.equal(panel.textContent, '节点: agentBEND');
assert.notEqual(panel.children[0], header);

// 阅读旧内容或正在选中文字时不强制滚到底
panel.scrollHeight = 1000; panel.clientHeight = 100; panel.scrollTop = 10;
a.push(text('more'));
NGO.renderThinking(panel, a, idA);
assert.equal(panel.scrollTop, 10);
panel.scrollTop = 900; selection = {isCollapsed: false};
a.push(text('selected'));
NGO.renderThinking(panel, a, idA);
assert.equal(panel.scrollTop, 900);
selection = {isCollapsed: true};
a.push(text('follow'));
NGO.renderThinking(panel, a, idA);
assert.equal(panel.scrollTop, 1000);

// ── 内联思考块：点号动画 / 折叠入口 / 清理 ──
function makeRow() {
  const row = new El();
  const target = new El('div');
  row.appendChild(target);
  row._contentTarget = target;
  return {row, target};
}

const {row, target} = makeRow();
let follows = 0;
NGO.thinking.begin(row, () => { follows += 1; });
assert.equal(follows, 1, '开始展开时窗口就要跟到底部');
const dots = target.firstChild;
assert.ok(dots);
assert.equal(dots.className, 'think-pending');
const block = row.children[0].children[1];
assert.equal(block.className, 'think-block');
assert.equal(block.getAttribute('data-state'), 'thinking');
const toggle = block.children[0], thinkPanel = block.children[1];
assert.equal(toggle.className, 'think-toggle');
assert.equal(toggle.textContent, '思考中');
// 思考中默认展开：必须能直接看见思考过程
assert.equal(thinkPanel.classList.contains('hidden'), false);
assert.equal(toggle.getAttribute('aria-expanded'), 'true');
assert.equal(block.getAttribute('data-expanded'), 'true');

// 点号 1→2→3→1 循环
assert.equal(dots.textContent, NGO.thinking.DOTS[0]);
clock.flush(400); assert.equal(dots.textContent, '..');
clock.flush(400); assert.equal(dots.textContent, '...');
clock.flush(400); assert.equal(dots.textContent, '.');

// 展开状态下思考边到边画，每次增量都要把窗口带到最底端
const identity = {};
NGO.thinking.update(row, [{kind: 'node', name: 'agent'}, text('想了一下')], identity);
assert.equal(thinkPanel.textContent, '节点: agent想了一下');
assert.equal(NGO.thinking.isExpanded(row), true);
assert.equal(NGO.thinking.traceOf(row).length, 2);
assert.equal(follows, 2, '每次思考增量都要跟随');
NGO.thinking.update(row, [{kind: 'node', name: 'agent'}, text('想了一下'), text('继续')], identity);
assert.equal(follows, 3);
// 折叠状态不渲染也不跟随，避免白做 DOM
const folded = makeRow();
let foldedFollows = 0;
NGO.thinking.begin(folded.row, () => { foldedFollows += 1; });
assert.equal(foldedFollows, 1);
NGO.thinking.toggle(folded.row);
assert.equal(NGO.thinking.isExpanded(folded.row), false);
NGO.thinking.update(folded.row, [text('看不见的思考')]);
assert.equal(foldedFollows, 1, '折叠状态不该跟随');
assert.equal(NGO.thinking.isExpanded(folded.row), false);
NGO.thinking.finish(folded.row);

// 回答开始出现 → 思考阶段结束，自动收起
NGO.thinking.collapseForAnswer(row);
assert.equal(NGO.thinking.isExpanded(row), false);
assert.equal(block.getAttribute('data-expanded'), 'false');
assert.equal(block.getAttribute('data-state'), 'thinking', '正文开始时思考标记还不该变 done');
// 回答正文会清空气泡，点号节点随之摘除，且不该再有空转定时器
target.textContent = '';
assert.equal(clock.jobs.size, 0);
NGO.thinking.finish(row);
assert.equal(block.getAttribute('data-state'), 'done');
assert.equal(toggle.textContent, '思考过程');
assert.equal(target.children.length, 0, '结束后气泡里不该还留着点号');
assert.equal(row._thinkDots, null);
assert.equal(NGO.thinking.isExpanded(row), false, '结束后保持常态折叠');

// 单击标记展开/收回，展开时才渲染
assert.equal(NGO.thinking.toggle(row), true);
assert.equal(NGO.thinking.isExpanded(row), true);
assert.equal(thinkPanel.textContent, '节点: agent想了一下继续');
assert.equal(follows, 4, '手动展开也要跟到底部');
assert.equal(toggle.getAttribute('aria-expanded'), 'true');
NGO.thinking.toggle(row);
assert.equal(NGO.thinking.isExpanded(row), false);
assert.equal(toggle.getAttribute('aria-expanded'), 'false');

// 展开状态下新思考继续追加，节点复用
NGO.thinking.toggle(row);
const reused = thinkPanel.children[1].children[0];
const trace = NGO.thinking.traceOf(row);
trace.push(text('又想了想'));
NGO.thinking.update(row, trace, identity);
assert.equal(thinkPanel.children[1].children[0], reused);
assert.equal(thinkPanel.textContent, '节点: agent想了一下继续又想了想');

// 没有思考内容时不留下空壳
const empty = makeRow();
NGO.thinking.begin(empty.row);
NGO.thinking.update(empty.row, []);
NGO.thinking.finish(empty.row);
assert.equal(empty.row.children.length, 1);
assert.equal(empty.row.children[0], empty.target);
assert.equal(empty.row._thinkBlock, null);

// 用户手动展开过，回答开始时不能被自动收起覆盖
const manual = makeRow();
NGO.thinking.begin(manual.row);
NGO.thinking.update(manual.row, [text('手动展开的思考')]);
NGO.thinking.toggle(manual.row);          // 先收起
assert.equal(NGO.thinking.isExpanded(manual.row), false);
NGO.thinking.toggle(manual.row);          // 再展开 → 用户已表达意愿
assert.equal(NGO.thinking.isExpanded(manual.row), true);
NGO.thinking.collapseForAnswer(manual.row);
assert.equal(NGO.thinking.isExpanded(manual.row), true, '用户手动展开后不该被自动收起');
NGO.thinking.finish(manual.row);
assert.equal(NGO.thinking.isExpanded(manual.row), true, 'finish 也不该覆盖用户选择');

// drop 要把气泡收回原位（失败/取消/换存档）
const dropped = makeRow();
NGO.thinking.begin(dropped.row);
NGO.thinking.update(dropped.row, [text('丢了')]);
dropped.row.remove();
NGO.thinking.drop(dropped.row);
assert.equal(dropped.row.children.length, 1);
assert.equal(dropped.row.children[0], dropped.target);
assert.equal(dropped.row._thinkBlock, null);

// clearAll 之后不能再有点号定时器
const pending = makeRow();
NGO.thinking.begin(pending.row);
assert.ok(clock.jobs.size > 0);
NGO.thinking.clearAll();
assert.equal(clock.jobs.size, 0);
assert.equal(pending.target.children.length, 0);

console.log('Thinking behavior passed: renderer identity/prefix/scroll plus inline block dots, toggle, reuse and cleanup');
