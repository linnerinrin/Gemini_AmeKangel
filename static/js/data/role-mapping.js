/**
 * access/data/role-mapping 角色→模式索引
 * 与 contracts/role2idx.json 保持同步
 * 索引范围 "0"–"11"，后端预留至 12 个 mode
 * 0 = jine(糖糖私聊)  1 = 超天酱直播  2 = 堕天酱直播  3 = 糖糖直播
 */
'use strict';

window.NGO = window.NGO || {};
NGO.role2idx = {
  ame:       "0",
  kangel:    "1",
  dangel:    "2",
  "ame-live": "3"
};
