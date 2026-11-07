import hashlib
import struct
import sys

from tokenizers import Tokenizer

CHUNK = 1 << 20

def main():
    model, path = sys.argv[1], sys.argv[2]
    t = Tokenizer.from_file(model)
    h = hashlib.md5()
    total = 0
    tail = b""
    with open(path, "rb") as f:
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            buf = tail + data
            cut = buf.rfind(b" ")
            if cut < 0:
                tail = buf
                continue
            piece, tail = buf[:cut + 1], buf[cut + 1:]
            ids = t.encode(piece.decode("utf-8")).ids
            h.update(struct.pack("<%dI" % len(ids), *ids))
            total += len(ids)
    if tail:
        ids = t.encode(tail.decode("utf-8")).ids
        h.update(struct.pack("<%dI" % len(ids), *ids))
        total += len(ids)
    print("%s %s %d tokens" % (model.split("/")[-1], h.hexdigest(), total))

if __name__ == "__main__":
    main()
