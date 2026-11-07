#!/usr/bin/env python3
"""Strip every comment from the C/C++ sources (the standing no-comment
rule). String and character literals are preserved. Line structure is
preserved so diagnostics keep their line numbers.

usage: strip_comments.py FILE...
"""
import sys

def strip(text):
    out = []
    i = 0
    n = len(text)
    line_start = True
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            q = c
            out.append(c)
            i += 1
            while i < n:
                d = text[i]
                out.append(d)
                i += 1
                if d == "\\" and i < n:
                    out.append(text[i])
                    i += 1
                    continue
                if d == q:
                    break
                if d == "\n":
                    break
            line_start = False
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                if text[i] == "\\" and i + 1 < n and text[i + 1] == "\n":
                    i += 2
                    out.append("\n")
                    continue
                i += 1
            while out and out[-1] in " \t":
                out.pop()
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and
                                 text[i + 1] == "/"):
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
            while out and out[-1] in " \t":
                out.pop()
            continue
        out.append(c)
        i += 1
    res = "".join(out)
    lines = []
    for ln in res.split("\n"):
        if ln.strip() == "":
            lines.append("")
        else:
            lines.append(ln.rstrip())
    return "\n".join(lines)

def main():
    for path in sys.argv[1:]:
        with open(path) as f:
            src = f.read()
        with open(path, "w") as f:
            f.write(strip(src))
        print("stripped " + path)

if __name__ == "__main__":
    main()
