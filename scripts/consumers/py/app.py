import tokensim
import time, resource

t = tokensim.tokenizer("/home/raising-reuel/tokenizer/models/cl100k_base.tiktoken",
                       "tiktoken", "cl100k")
data = open("/home/raising-reuel/tokenizer/testdata/owt.txt", "rb").read()
print(f"input: {len(data) // (1024*1024)} MB")
print(f"rss after load+read: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} KB")
mb = len(data) / 1e6

t0 = time.perf_counter()
raw = t(data)
t1 = time.perf_counter()
print(f"1t tok(data): {mb/(t1-t0):.1f} MB/s ({len(raw)//4} ids)")

t0 = time.perf_counter()
raw = t.encode(data)
t1 = time.perf_counter()
print(f"1t encode warm: {mb/(t1-t0):.1f} MB/s ({len(raw)//4} ids)")

t0 = time.perf_counter()
raw6 = t.encode_threads(data, 6)
t1 = time.perf_counter()
print(f"6t encode_threads: {mb/(t1-t0):.1f} MB/s ({len(raw6)//4} ids)")

bound = t.encode_bound(len(data))
buf = bytearray(bound)
t0 = time.perf_counter()
n16 = t.encode_threads_into(data, buf, 16)
t1 = time.perf_counter()
print(f"16t into: {mb/(t1-t0):.1f} MB/s ({n16} ids)")
del buf

rt = t.decode(raw)
print(f"round-trip: {rt == data}")
print(f"rss after encode: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} KB")
