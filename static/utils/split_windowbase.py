"""
Split windowbase_active.png and windowbase_inactive.png into 3 vertical parts:
  - header: top 20% (y=0 ~ 40)
  - body:   mid 70% (y=40 ~ 180)
  - footer: bot 10% (y=180 ~ 200)

Horizontal: 4% border on each side (12px), content area is 92% (276px).
"""

import os
from PIL import Image

SRC = r'D:\games\Needy.Streamer.Overload.v1.4.0(Danjipai.com)\content'

names = ['windowbase_active', 'windowbase_inactive']

for name in names:
    img = Image.open(os.path.join(SRC, name + '.png'))
    w, h = img.size  # 300, 200

    header = img.crop((0, 0, w, 40))
    body   = img.crop((0, 40, w, 180))
    footer = img.crop((0, 180, w, 200))

    header.save(os.path.join(SRC, name + '_header.png'))
    body.save(os.path.join(SRC, name + '_body.png'))
    footer.save(os.path.join(SRC, name + '_footer.png'))

    print(f'{name}: header {header.size}, body {body.size}, footer {footer.size}')

print('Done.')
