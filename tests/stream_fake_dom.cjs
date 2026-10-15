const assert = require('node:assert/strict');
class El {
 constructor(tag='div') { this.tag=tag;this.nodeType=tag==='#text'?3:1;this.childNodes=[];this.parentNode=null;this._text='';this.style={};this._scrollTop=0;this._listeners={};this.scrollHeight=100;this.clientHeight=100;this.attrs={};this.classes=new Set();this.classList={add:(...xs)=>xs.forEach(x=>this.classes.add(x)),remove:(...xs)=>xs.forEach(x=>this.classes.delete(x)),contains:x=>this.classes.has(x),toggle:x=>this.classes.has(x)?this.classes.delete(x):this.classes.add(x)}; }
 // 真实 DOM 里改 scrollTop 会派发 scroll 事件，替身也要派发，
 // 否则"用户往上滚 → 停止跟随"这类依赖事件的逻辑在测试里永远测不到。
 get scrollTop(){return this._scrollTop||0;}
 set scrollTop(v){const next=Number(v)||0;if(next===(this._scrollTop||0))return;this._scrollTop=next;const ls=this._listeners.scroll;if(ls)ls.slice().forEach(fn=>fn({type:'scroll',target:this}));}
 get children(){return this.childNodes;}
 get lastChild(){return this.childNodes.at(-1)||null;}
 get firstChild(){return this.childNodes[0]||null;}
 get parentElement(){return this.parentNode;}
 // 真实 DOM 里 className 与 classList 是同一份数据，替身也必须一致，
 // 否则生产代码用 className 设类、测试用 classList 判断就会假失败。
 get className(){return [...this.classes].join(' ');}
 set className(v){this.classes.clear();String(v||'').split(/\s+/).filter(Boolean).forEach(x=>this.classes.add(x));}
 contains(node){for(let n=node;n;n=n.parentNode)if(n===this)return true;return false;}
 set textContent(v){this.childNodes.forEach(c=>c.parentNode=null);this.childNodes=[];this._text=String(v);}
 get textContent(){return this._text+this.childNodes.map(c=>c.textContent).join('');}
 get data(){return this._text;} set data(v){this._text=String(v);}
 appendData(v){this._text+=v;} deleteData(start,count){this._text=this._text.slice(0,start)+this._text.slice(start+count);}
 appendChild(c){if(c.parentNode)c.remove();c.parentNode=this;this.childNodes.push(c);return c;}
 removeChild(c){const i=this.childNodes.indexOf(c);assert.ok(i>=0);this.childNodes.splice(i,1);c.parentNode=null;return c;}
 replaceChild(c,old){const i=this.childNodes.indexOf(old);assert.ok(i>=0);this.childNodes[i]=c;c.parentNode=this;old.parentNode=null;}
 remove(){if(this.parentNode)this.parentNode.removeChild(this);}
 addEventListener(type,fn){(this._listeners[type]=this._listeners[type]||[]).push(fn);} setAttribute(k,v){this.attrs[k]=v;} getAttribute(k){return this.attrs[k];}
 querySelector(){return null;} querySelectorAll(){return [];}
}
function timers(){let id=0;const jobs=new Map();return {jobs,setTimeout(fn,delay){const n=++id;jobs.set(n,{fn,delay});return n;},clearTimeout(n){jobs.delete(n);},flush(max=Infinity){for(const [n,j] of [...jobs])if(j.delay<=max){jobs.delete(n);j.fn();}}};}
module.exports={El,timers};
