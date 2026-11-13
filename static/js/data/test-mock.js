/**
 * access/test-mock — ?test 模式下拦截 apiJson 返回 mock 数据
 * 用法: 在 URL 后加 ?test 即可，如 http://localhost:8000/desktop.html?test
 */
(function() {
  'use strict';

  if (window.location.search.indexOf('test') === -1) return;

  var NGO = window.NGO = window.NGO || {};
  var _realApiJson = NGO.apiJson;

  var MOCK = {
    '/chat/sessions':          { sessions: [] },
    '/chat/new_session':       { ok: true },
    '/chat/chose_session':     { ok: true },
    '/chat/delete_session':    { ok: true },
    '/chat/history':           [],
    '/chat/get_status':        { followers: 0, stress: 0, affection: 50, mental: 0 },
    '/chat/mode_change':       { ok: true }
  };

  function matchMock(url) {
    for (var path in MOCK) {
      if (url.indexOf(path) !== -1 && (url.length === path.length || url.charAt(path.length) === '?')) {
        return MOCK[path];
      }
    }
    return null;
  }

  NGO.apiJson = function(url, options) {
    var mock = matchMock(url);
    if (mock) return Promise.resolve(mock);
    if (url.indexOf('/chat/single_chat') !== -1) return Promise.reject(new Error('Mock: use fallback'));
    if (_realApiJson) return _realApiJson(url, options);
    return Promise.reject(new Error('No mock for: ' + url));
  };

  console.log('[test-mock] 测试模式已启用');
})();
