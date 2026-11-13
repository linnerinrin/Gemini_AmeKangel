/*
settings — 模型设置接口（P5a：只做 OpenAI 兼容厂商）
后端只回掩码后的 key；api_key 只在用户真的改过输入框时才提交。
*/
(function() {
  'use strict';

  var NGO = window.NGO = window.NGO || {};

  NGO.settingsAPI = {
    // GET：active 厂商 / 各厂商 base_url+model / 掩码 key / 是否已配置可用 key
    getSettings: function() {
      return NGO.apiJson('/chat/settings/llm', { method: 'GET' });
    },
    // POST：保存并切换 active；401/409/422 由 apiJson 抛出 detail
    saveSettings: function(active, profiles) {
      return NGO.apiJson('/chat/settings/llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: active, profiles: profiles })
      });
    },
    // POST：用候选配置做一次最小连通性测试，不回显 key
    testSettings: function(provider, baseUrl, model, apiKey) {
      // 可选字段只在用户真的填/改过时才提交（api_key 尤其不能带掩码或空值）；
      // provider 写在字面量里，契约测试据此校验该路径的请求形状
      var extra = {};
      if (baseUrl) extra.base_url = baseUrl;
      if (model) extra.model = model;
      if (apiKey) extra.api_key = apiKey;
      return NGO.apiJson('/chat/settings/llm/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: provider, ...extra })
      });
    }
  };
})();
