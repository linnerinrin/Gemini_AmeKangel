/*


*/
(function() {
  'use strict';

  var NGO = window.NGO = window.NGO || {};

  NGO.get_status = {
    getStatus: function() {
      return NGO.apiJson('/chat/get_status', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ session_id: NGO.currentSessionId || null }) });
    }
  };
})();
