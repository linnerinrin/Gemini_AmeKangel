(function() {
  'use strict';
  var NGO = window.NGO = window.NGO || {};
  NGO.chatAPI = {
    singleChat: function(prompt, mode) {
      return NGO.apiJson('/chat/single_chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({prompt: prompt, session_id: NGO.currentSessionId || null, mode: mode == null ? null : String(mode)})});
    },
    streamChat: async function(prompt, mode, onEvent, signal) {
      var origin = window.location.port === '63342' ? window.location.protocol + '//localhost:5577' : window.location.origin;
      var response = await fetch(origin + '/chat/stream_chat', {method: 'POST', signal: signal, headers: {'Content-Type': 'application/json', Accept: 'text/event-stream'}, body: JSON.stringify({prompt: prompt, session_id: NGO.currentSessionId || null, mode: mode == null ? null : String(mode)})});
      if (!response.ok) { var error = await response.json().catch(function() {return {};}); throw new Error(error.detail || '请求失败 (' + response.status + ')'); }
      if (!response.body || !(response.headers.get('content-type') || '').includes('text/event-stream')) throw new Error('无效的流式响应');
      var reader = response.body.getReader(), decoder = new TextDecoder(), buffer = '', completed = false;
      try {
        while (true) {
          var part = await reader.read();
          buffer += decoder.decode(part.value, {stream: !part.done});
          var match;
          while ((match = /\r?\n\r?\n/.exec(buffer))) {
            var block = buffer.slice(0, match.index); buffer = buffer.slice(match.index + match[0].length);
            var data = block.split(/\r?\n/).filter(function(line) {return line.startsWith('data:');}).map(function(line) {return line.slice(5).replace(/^ /, '');}).join('\n');
            if (!data) continue;
            var event = JSON.parse(data);
            if (event.type === 'error') throw new Error(event.message || '生成失败');
            if (completed) throw new Error('完成后仍收到数据');
            onEvent(event);
            if (event.type === 'done') completed = true;
          }
          if (part.done) break;
        }
        if (!completed) throw new Error('连接中断，回答尚未完成；请手动重试');
      } finally { await reader.cancel().catch(function() {}); reader.releaseLock(); }
    },
    getHistory: function(mode) {
      // 路径写成字面量：契约测试按 apiJson('<path>' 校验"已登记路径确实被前端调用"
      var params = [];
      if (mode !== undefined && mode !== null && mode !== '') params.push('mode=' + encodeURIComponent(mode));
      if (NGO.currentSessionId) params.push('session_id=' + encodeURIComponent(NGO.currentSessionId));
      var query = params.length ? '?' + params.join('&') : '';
      return NGO.apiJson('/chat/history' + query, {method: 'GET'});
    }
  };
})();
