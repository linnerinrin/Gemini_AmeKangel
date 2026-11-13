/*
live 动画流
 */
'use strict';

window.NGO = window.NGO || {};

NGO.liveKangelFrames = {
  idle: [
    "stream_cho_idle_000.png",
    "stream_cho_idle_001.png"
  ],
  think: [
    "stream_cho_eeto_000.png",
    "stream_cho_eeto_001.png"
  ],
  smile: [
    "stream_cho_smile_001.png",
    "stream_cho_smile_002.png",
    "stream_cho_smile_003.png",
    "stream_cho_smile_004.png",
    "stream_cho_smile_005.png",
    "stream_cho_smile_006.png"
  ],
  shobon: [
    "stream_cho_shobon_000.png",
    "stream_cho_shobon_001.png"
  ],
  start: [
    "stream_cho_angel_000.png",
    "stream_cho_angel_001.png"
  ],
  pout: [
    "stream_cho_pout_000.png",
    "stream_cho_pout_001.png"
  ]
};

NGO.liveDangelFrames = {
  idle: [
    "stream_cho_b_idle_000.png",
    "stream_cho_b_idle_001.png"
  ],
  think: [
    "stream_cho_b_idle2_000.png",
    "stream_cho_b_idle2_001.png"
  ],
  smile: [
    "stream_cho_b_lower1_000.png",
    "stream_cho_b_lower1_001.png"
  ],
  shobon: [
    "stream_cho_b_lower2_000.png",
    "stream_cho_b_lower2_001.png"
  ],
  start: [
    "stream_cho_b_angle_000.png",
    "stream_cho_b_angle_001.png"
  ]
};

NGO.liveAmeFrames = {
  idle: [
    "stream_ame_idle_normal_000.png",
    "stream_ame_idle_normal_001.png",
    "stream_ame_idle_normal_002.png",
    "stream_ame_idle_normal_003.png",
    "stream_ame_idle_normal_004.png",
    "stream_ame_idle_normal_005.png",
    "stream_ame_idle_normal_006.png",
    "stream_ame_idle_normal_007.png",
    "stream_ame_idle_normal_008.png",
    "stream_ame_idle_normal_009.png",
    "stream_ame_idle_normal_010.png",
    "stream_ame_idle_normal_011.png"
  ],
  think: [
    "stream_ame_talk_000.png",
    "stream_ame_talk_001.png",
    "stream_ame_talk_002.png",
    "stream_ame_talk_003.png",
    "stream_ame_talk_004.png"
  ],
  smile: [
    "stream_ame_smile_000.png",
    "stream_ame_smile_001.png",
    "stream_ame_smile_002.png"
  ],
  shobon: [
    "stream_ame_negative_000.png",
    "stream_ame_negative_001.png",
    "stream_ame_negative_002.png",
    "stream_ame_negative_003.png",
    "stream_ame_negative_004.png",
    "stream_ame_negative_005.png",
    "stream_ame_negative_006.png",
    "stream_ame_negative_007.png"
  ],
  start: [
    "stream_ame_yanda_carisma_000.png",
    "stream_ame_yanda_carisma_001.png",
    "stream_ame_yanda_carisma_002.png",
    "stream_ame_yanda_carisma_003.png",
    "stream_ame_yanda_carisma_004.png",
    "stream_ame_yanda_carisma_005.png",
    "stream_ame_yanda_carisma_006.png",
    "stream_ame_yanda_carisma_007.png",
    "stream_ame_yanda_carisma_008.png",
    "stream_ame_yanda_carisma_009.png",
    "stream_ame_yanda_carisma_010.png",
    "stream_ame_yanda_carisma_011.png",
    "stream_ame_yanda_carisma_012.png"
  ]
};

// 若 jine 帧表已先加载，则同样补齐直播 ame 的状态待机键（顺序无关，两处互相补一次）。
if (window.NGO && window.NGO.jineAmeFrames) {
  for (var _jk in window.NGO.jineAmeFrames) {
    if (!Object.prototype.hasOwnProperty.call(window.NGO.liveAmeFrames, _jk)) {
      window.NGO.liveAmeFrames[_jk] = window.NGO.jineAmeFrames[_jk];
    }
  }
}
