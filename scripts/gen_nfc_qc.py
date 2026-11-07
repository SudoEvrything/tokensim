import unicodedata

trig = {}
for cp in range(0x80, 0x110000):
    ch = chr(cp)
    if 0xAC00 <= cp <= 0xD7A3: continue
    if unicodedata.normalize('NFC', ch) != ch: t = True
    elif unicodedata.combining(ch) != 0: t = True
    elif 0x1161 <= cp <= 0x1175 or 0x11A8 <= cp <= 0x11C2: t = True
    else: t = False
    if not t: continue
    u = ch.encode('utf-8')
    if len(u) < 2: continue
    trig.setdefault((u[0], u[1]), set())
    if len(u) > 2: trig[(u[0], u[1])].add(u[2])
