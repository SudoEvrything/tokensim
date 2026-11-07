"""Patch a sentencepiece model so its TrainerSpec carries a custom
unk_surface (field 44). Used to exercise the parser and the decode path that
would otherwise never be reached: every model in the tree uses the protocol
default " \u2047 ".

usage: make_unk_surface_fixture.py IN.model OUT.model SURFACE
"""
import sys

def read_varint(b, i):
    v = 0
    s = 0
    while True:
        x = b[i]
        i += 1
        v |= (x & 0x7F) << s
        if x < 0x80:
            return v, i
        s += 7

def write_varint(v):
    out = bytearray()
    while True:
        x = v & 0x7F
        v >>= 7
        if v:
            out.append(x | 0x80)
        else:
            out.append(x)
            return bytes(out)

def main():
    src, dst, surface = sys.argv[1], sys.argv[2], sys.argv[3]
    b = open(src, "rb").read()
    i = 0
    spec = None
    while i < len(b):
        tag, j = read_varint(b, i)
        field, wire = tag >> 3, tag & 7
        if wire != 2:
            raise SystemExit("unexpected wire type %d at %d" % (wire, i))
        ln, k = read_varint(b, j)
        if field == 2:
            spec = (i, j, k, k + ln)
        i = k + ln
    if spec is None:
        raise SystemExit("no TrainerSpec (field 2) in %s" % src)
    tag_i, len_i, body_i, body_end = spec
    payload = surface.encode("utf-8")
    extra = write_varint((44 << 3) | 2) + write_varint(len(payload)) + payload
    body = b[body_i:body_end] + extra
    out = bytearray()
    out += b[:tag_i] + write_varint((2 << 3) | 2) + write_varint(len(body))
    out += body + b[body_end:]
    open(dst, "wb").write(bytes(out))
    print("wrote %s: TrainerSpec %d -> %d bytes, unk_surface=%r"
          % (dst, body_end - body_i, len(body), surface))

if __name__ == "__main__":
    main()
