/*
session api

api request response function

sessions [] [] 拿到会话列表
new_session [session_num] [session_id] 新启会话
chose_session [session_num] [session_id] 选择会话
delete_session [session_num] [] 删除会话
 */
(function() {
  'use strict';

  var NGO = window.NGO = window.NGO || {};

  NGO.saveAPI = {
    getSessions: function() {
      return NGO.apiJson('/chat/sessions', { method: 'GET' });
    },

    newSession: function(sessionNum) {
      return NGO.apiJson('/chat/new_session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_num: sessionNum })
      });
    },

    chooseSession: function(sessionNum) {
      return NGO.apiJson('/chat/chose_session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_num: sessionNum })
      });
    },

    deleteSession: function(sessionNum) {
      return NGO.apiJson('/chat/delete_session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_num: sessionNum })
      });
    }
  };
})();
