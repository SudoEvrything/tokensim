import sys

sys.path.insert(0, "" + __import__("site").getusersitepackages() + "")
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="models/reference/t5_spiece.model")
for name in ("t5_fuzz_mismatch", "t5_tie_negative"):
    src = "tests/fixtures/%s.bin" % name
    ids = sp.encode(open(src, "rb").read(), out_type=int)
    with open("tests/fixtures/%s.expected" % name, "w") as w:
        w.write(" ".join(map(str, ids)) + "\n")
    print(name, len(ids))
