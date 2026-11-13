/*

 */
(function() {
  'use strict';

  var NGO = window.NGO = window.NGO || {};

  NGO.modeAPI = {
    changeMode: function(mode) {
      return NGO.apiJson('/chat/mode_change', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode, session_id: NGO.currentSessionId || null })
      });
    }
  };
})();
