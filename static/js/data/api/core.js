/*
core
抽象api方法 apiJson
*/
var BASE = '/static';

(function() {
  'use strict';

  var NGO = window.NGO = window.NGO || {};
  // 当前存档的session_id，由存档进入/新建流程写入；会话类接口显式携带，后端按请求绑定
  NGO.currentSessionId = null;
  var API_ORIGIN = window.location.port === '63342'
    ? window.location.protocol + '//localhost:5577'
    : window.location.origin;

  NGO.apiJson =function(url, options) {
    options = options || {};
    var sourceHeaders = options.headers || {};
    var headers={...sourceHeaders};
    options.headers = headers;

    var requestUrl = url.indexOf('/') === 0 ? API_ORIGIN + url : url;
    return fetch(requestUrl,options).then(function(response) {
      if (!response.ok) {
        return response.json().then(function(err) {
          throw new Error(err.detail || err.message || '请求失败');
        });
      }
      return response.json();
    })
  };
})();
