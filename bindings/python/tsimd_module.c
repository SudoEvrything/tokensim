#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "tokenizer.h"
#include "batch.h"
#include "model_internal.h"
#include "tokensimd_version.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>

typedef struct {
    PyObject_HEAD
    tsimd_tokenizer_t *tok;
    pthread_mutex_t mu;
    uint32_t *scratch;
    size_t scratch_cap;
    uint8_t *dec_scratch;
    size_t dec_scratch_cap;
} TokObject;

static PyObject *TokError;

static void Tok_dealloc(TokObject *self) {
    if (self->tok) {
        pthread_mutex_destroy(&self->mu);
        tsimd_tokenizer_free(self->tok);
        free(self->tok);
    }
    free(self->scratch);
    free(self->dec_scratch);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int tok_scratch_reserve(TokObject *self, size_t len) {
    size_t bound = tsimd_tokenizer_encode_bound(self->tok, len);
    if (self->scratch_cap >= bound) return 0;
    size_t cap = self->scratch_cap ? self->scratch_cap : 4096;
    while (cap < bound) cap *= 2;
    uint32_t *grown =
        realloc(self->scratch, cap * sizeof(uint32_t));
    if (!grown) return -1;
    self->scratch = grown;
    self->scratch_cap = cap;
    return 0;
}

static PyObject *tok_scratch_result(TokObject *self, size_t count) {
    return PyBytes_FromStringAndSize((const char *)self->scratch,
                                     (Py_ssize_t)count * 4);
}

static int format_from_obj(PyObject *obj, const char *path,
                           tsimd_model_format_t *out) {
    if (obj == NULL || obj == Py_None) {
        const char *dot = path ? strrchr(path, '.') : NULL;
        if (dot && strcmp(dot, ".tiktoken") == 0)
            *out = TSIMD_MODEL_TIKTOKEN;
        else if (dot && strcmp(dot, ".model") == 0)
            *out = TSIMD_MODEL_SENTENCEPIECE;
        else
            *out = TSIMD_MODEL_TOKENIZER_JSON;
        return 0;
    }
    const char *s = PyUnicode_AsUTF8(obj);
    if (!s) return -1;
    if (strcmp(s, "tokenizer_json") == 0) *out = TSIMD_MODEL_TOKENIZER_JSON;
    else if (strcmp(s, "sentencepiece") == 0) *out = TSIMD_MODEL_SENTENCEPIECE;
    else if (strcmp(s, "tiktoken") == 0) *out = TSIMD_MODEL_TIKTOKEN;
    else if (strcmp(s, "gpt2") == 0) *out = TSIMD_MODEL_GPT2;
    else {
        PyErr_Format(PyExc_ValueError, "unknown format: %s", s);
        return -1;
    }
    return 0;
}

static PyObject *tok_encode(TokObject *self, PyObject *args) {
    Py_buffer view;
    if (!PyArg_ParseTuple(args, "y*", &view)) return NULL;
    size_t count = 0;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&self->mu);
    rc = tok_scratch_reserve(self, (size_t)view.len);
    if (rc == 0)
        rc = tsimd_tokenizer_encode_parallel_into(
            self->tok, (const uint8_t *)view.buf, (size_t)view.len, 0,
            self->scratch, self->scratch_cap, &count);
    pthread_mutex_unlock(&self->mu);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (rc != 0) {
        PyErr_SetString(TokError, "encode failed");
        return NULL;
    }
    return tok_scratch_result(self, count);
}

static PyObject *tok_call(TokObject *self, PyObject *args, PyObject *kwargs) {
    if (kwargs && PyDict_Size(kwargs) > 0) {
        PyErr_SetString(PyExc_TypeError,
                        "tok(data) takes no keyword arguments");
        return NULL;
    }
    return tok_encode(self, args);
}

static PyObject *tok_encode_list(TokObject *self, PyObject *args) {
    Py_buffer view;
    if (!PyArg_ParseTuple(args, "y*", &view)) return NULL;
    size_t count = 0;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&self->mu);
    rc = tok_scratch_reserve(self, (size_t)view.len);
    if (rc == 0)
        rc = tsimd_tokenizer_encode_parallel_into(
            self->tok, (const uint8_t *)view.buf, (size_t)view.len, 0,
            self->scratch, self->scratch_cap, &count);
    pthread_mutex_unlock(&self->mu);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (rc != 0) {
        PyErr_SetString(TokError, "encode failed");
        return NULL;
    }
    PyObject *list = PyList_New((Py_ssize_t)count);
    if (!list) return NULL;
    for (size_t i = 0; i < count; i++) {
        PyObject *v = PyLong_FromUnsignedLong(self->scratch[i]);
        if (!v || PyList_SetItem(list, (Py_ssize_t)i, v) != 0) {
            Py_XDECREF(v);
            Py_DECREF(list);
            return NULL;
        }
    }
    return list;
}

static PyObject *tok_encode_bytes(TokObject *self, PyObject *args) {
    Py_buffer view;
    if (!PyArg_ParseTuple(args, "y*", &view)) return NULL;
    size_t count = 0;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&self->mu);
    rc = tok_scratch_reserve(self, (size_t)view.len);
    if (rc == 0)
        rc = tsimd_tokenizer_encode_parallel_into(
            self->tok, (const uint8_t *)view.buf, (size_t)view.len, 0,
            self->scratch, self->scratch_cap, &count);
    pthread_mutex_unlock(&self->mu);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (rc != 0) {
        PyErr_SetString(TokError, "encode failed");
        return NULL;
    }
    return tok_scratch_result(self, count);
}

static PyObject *tok_encode_threads(TokObject *self, PyObject *args) {
    Py_buffer view;
    int threads = 0;
    if (!PyArg_ParseTuple(args, "y*|i", &view, &threads)) return NULL;
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    int rc;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&self->mu);
    rc = tsimd_tokenizer_encode_parallel(self->tok,
                                         (const uint8_t *)view.buf,
                                         (size_t)view.len, threads, &seq);
    pthread_mutex_unlock(&self->mu);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (rc != 0) {
        PyErr_SetString(TokError, "encode failed");
        return NULL;
    }
    PyObject *out = PyBytes_FromStringAndSize((const char *)seq.ids,
                                              (Py_ssize_t)seq.count * 4);
    tsimd_token_sequence_free(&seq);
    return out;
}

static PyObject *tok_encode_batch(TokObject *self, PyObject *args) {
    PyObject *docs_obj;
    int threads = 0;
    if (!PyArg_ParseTuple(args, "O|i", &docs_obj, &threads)) return NULL;
    PyObject *fast = PySequence_Fast(docs_obj,
                                     "encode_batch expects a sequence of "
                                     "bytes-like objects");
    if (!fast) return NULL;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n == 0) {
        Py_DECREF(fast);
        return PyList_New(0);
    }
    Py_buffer *views = calloc((size_t)n, sizeof(Py_buffer));
    tsimd_input_span_t *spans = malloc((size_t)n * sizeof *spans);
    if (!views || !spans) {
        free(views);
        free(spans);
        Py_DECREF(fast);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        if (PyObject_GetBuffer(item, &views[i], PyBUF_CONTIG_RO) != 0) {
            for (Py_ssize_t j = 0; j < i; j++) PyBuffer_Release(&views[j]);
            free(views);
            free(spans);
            Py_DECREF(fast);
            return NULL;
        }
        spans[i].data = (const uint8_t *)views[i].buf;
        spans[i].length = (size_t)views[i].len;
    }
    tsimd_token_batch_t batch;
    memset(&batch, 0, sizeof batch);
    int rc;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&self->mu);
    rc = tsimd_tokenizer_encode_batch(self->tok, spans, (size_t)n, &batch,
                                      threads);
    pthread_mutex_unlock(&self->mu);
    Py_END_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; i++) PyBuffer_Release(&views[i]);
    free(views);
    free(spans);
    Py_DECREF(fast);
    if (rc != 0) {
        PyErr_SetString(TokError, "encode_batch failed");
        return NULL;
    }
    PyObject *out = PyList_New(n);
    if (!out) {
        tsimd_token_batch_free(&batch);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        size_t start = batch.offsets[i];
        size_t end = i + 1 < n ? batch.offsets[i + 1] : batch.id_count;
        PyObject *piece = PyBytes_FromStringAndSize(
            (const char *)(batch.ids + start), (Py_ssize_t)(end - start) * 4);
        if (!piece || PyList_SetItem(out, i, piece) != 0) {
            Py_XDECREF(piece);
            Py_DECREF(out);
            tsimd_token_batch_free(&batch);
            return NULL;
        }
    }
    tsimd_token_batch_free(&batch);
    return out;
}

static PyObject *tok_decode_into(TokObject *self, PyObject *args) {
    PyObject *ids_obj, *out_obj;
    int skip_special = 0;
    int threads = 0;
    if (!PyArg_ParseTuple(args, "OO|pi", &ids_obj, &out_obj, &skip_special,
                          &threads))
        return NULL;
    Py_buffer iv, ov;
    if (PyObject_GetBuffer(ids_obj, &iv, PyBUF_CONTIG_RO) != 0) return NULL;
    if (PyObject_GetBuffer(out_obj, &ov, PyBUF_WRITABLE) != 0) {
        PyBuffer_Release(&iv);
        return NULL;
    }
    if (iv.len % 4 != 0) {
        PyBuffer_Release(&iv);
        PyBuffer_Release(&ov);
        PyErr_SetString(TokError,
                        "decode ids length must be a multiple of 4");
        return NULL;
    }
    uint32_t *ids = NULL;
    int ids_copy = 0;
    if (((uintptr_t)iv.buf & 3) == 0) {
        ids = (uint32_t *)iv.buf;
    } else {
        ids = malloc((size_t)(iv.len / 4) * sizeof(uint32_t));
        if (!ids) {
            PyBuffer_Release(&iv);
            PyBuffer_Release(&ov);
            return PyErr_NoMemory();
        }
        memcpy(ids, iv.buf, iv.len);
        ids_copy = 1;
    }
    size_t got = 0;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&self->mu);
    if ((size_t)(iv.len / 4) >= (1u << 18))
        rc = tsimd_tokenizer_decode_parallel(self->tok, ids,
                                             (size_t)(iv.len / 4),
                                             skip_special,
                                             ov.len ? ov.buf : NULL, ov.len,
                                             &got, threads);
    else
        rc = tsimd_tokenizer_decode_into(self->tok, ids,
                                         (size_t)(iv.len / 4), skip_special,
                                         ov.buf, ov.len, &got);
    pthread_mutex_unlock(&self->mu);
    Py_END_ALLOW_THREADS
    if (ids_copy) free(ids);
    PyBuffer_Release(&iv);
    PyBuffer_Release(&ov);
    if (rc != 0 || got == 0) {
        if (rc == 0) {
            PyErr_SetString(TokError, "decode_into produced nothing");
        } else {
            PyErr_SetString(TokError, "decode_into failed");
        }
        return NULL;
    }
    return PyLong_FromSize_t(got);
}

static PyObject *tok_encode_bound(TokObject *self, PyObject *args) {
    Py_ssize_t len;
    if (!PyArg_ParseTuple(args, "n", &len)) return NULL;
    if (len < 0) {
        PyErr_SetString(TokError, "length must be non-negative");
        return NULL;
    }
    size_t bound = tsimd_tokenizer_encode_bound(self->tok, (size_t)len);
    return PyLong_FromSize_t(bound * 4);
}

static PyObject *tok_encode_threads_into(TokObject *self, PyObject *args) {
    Py_buffer view;
    PyObject *out_obj;
    int threads = 0;
    if (!PyArg_ParseTuple(args, "y*O|i", &view, &out_obj, &threads))
        return NULL;
    Py_buffer ov;
    if (PyObject_GetBuffer(out_obj, &ov, PyBUF_WRITABLE) != 0) {
        PyBuffer_Release(&view);
        return NULL;
    }
    if (ov.len % 4 != 0) {
        PyBuffer_Release(&ov);
        PyBuffer_Release(&view);
        PyErr_SetString(TokError, "out buffer length must be a multiple of 4");
        return NULL;
    }
    size_t cap = (size_t)ov.len / 4;
    size_t count = 0;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&self->mu);
    rc = tsimd_tokenizer_encode_parallel_into(self->tok,
                                              (const uint8_t *)view.buf,
                                              (size_t)view.len, threads,
                                              (uint32_t *)ov.buf, cap,
                                              &count);
    pthread_mutex_unlock(&self->mu);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&ov);
    PyBuffer_Release(&view);
    if (rc != 0) {
        PyErr_SetString(TokError, "encode failed");
        return NULL;
    }
    return PyLong_FromSize_t(count);
}

typedef struct {
    PyObject *obj;
    char *buf;
    size_t total;
} py_dec_alloc_ud_t;

static void *py_dec_alloc(void *ud, size_t total) {
    py_dec_alloc_ud_t *u = ud;
    PyGILState_STATE st = PyGILState_Ensure();
    u->obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total);
    u->buf = u->obj ? PyBytes_AS_STRING(u->obj) : NULL;
    u->total = total;
    PyGILState_Release(st);
    return u->buf;
}

static PyObject *tok_decode(TokObject *self, PyObject *args) {    PyObject *obj;
    int skip_special = 0;
    if (!PyArg_ParseTuple(args, "O|p", &obj, &skip_special)) return NULL;
    uint32_t *ids = NULL;
    Py_ssize_t n = 0;
    int have_view = 0;
    Py_buffer view;
    memset(&view, 0, sizeof view);
    if (PyList_Check(obj)) {
        n = PyList_Size(obj);
        ids = malloc((size_t)(n > 0 ? n : 1) * sizeof(uint32_t));
        if (!ids) return PyErr_NoMemory();
        for (Py_ssize_t i = 0; i < n; i++) {
            unsigned long v = PyLong_AsUnsignedLong(PyList_GetItem(obj, i));
            if (v == (unsigned long)-1 && PyErr_Occurred()) {
                free(ids);
                return NULL;
            }
            ids[i] = (uint32_t)v;
        }
    } else if (PyObject_GetBuffer(obj, &view, PyBUF_CONTIG_RO) == 0) {
        if (view.len % 4 != 0) {
            PyBuffer_Release(&view);
            PyErr_SetString(TokError,
                            "decode buffer length must be a multiple of 4");
            return NULL;
        }
        n = view.len / 4;
        if (((uintptr_t)view.buf & 3) == 0) {
            ids = view.buf;
            have_view = 1;
        } else {
            ids = malloc((size_t)(n > 0 ? n : 1) * sizeof(uint32_t));
            if (!ids) {
                PyBuffer_Release(&view);
                return PyErr_NoMemory();
            }
            memcpy(ids, view.buf, (size_t)n * 4);
            have_view = 0;
        }
    } else {
        PyErr_SetString(PyExc_TypeError,
                        "decode expects a list of ids or a bytes-like buffer");
        return NULL;
    }
    size_t got = 0;
    int rc = 0;
    if (n >= (1u << 18)) {
        py_dec_alloc_ud_t ud = {NULL, NULL, 0};
        Py_BEGIN_ALLOW_THREADS
        pthread_mutex_lock(&self->mu);
        rc = tsimd_decode_parallel_alloc_cb(self->tok, ids, (size_t)n,
                                            skip_special, py_dec_alloc, &ud,
                                            &got, 0);
        pthread_mutex_unlock(&self->mu);
        Py_END_ALLOW_THREADS
        if (rc == 0 && ud.obj && got > 0 && got <= ud.total) {
            if (got < ud.total &&
                _PyBytes_Resize(&ud.obj, (Py_ssize_t)got) != 0) {
                ud.obj = NULL;
                if (!have_view) free(ids);
                PyBuffer_Release(&view);
                return NULL;
            }
            if (!have_view) free(ids);
            PyBuffer_Release(&view);
            return ud.obj;
        }
        if (ud.obj) Py_DECREF(ud.obj);
        if (rc != 0 || got == 0) {
            if (!have_view) free(ids);
            PyBuffer_Release(&view);
            PyErr_SetString(TokError, "decode failed");
            return NULL;
        }
        got = 0;
    } else {
        Py_BEGIN_ALLOW_THREADS
        pthread_mutex_lock(&self->mu);
        rc = tsimd_tokenizer_decode_into(self->tok, ids, (size_t)n,
                                         skip_special, self->dec_scratch,
                                         self->dec_scratch_cap, &got);
        if (rc != 0 || got > self->dec_scratch_cap) {
            if (got > self->dec_scratch_cap) {
                uint8_t *grown = realloc(self->dec_scratch, got);
                if (grown) {
                    self->dec_scratch = grown;
                    self->dec_scratch_cap = got;
                }
            }
            if (self->dec_scratch && self->dec_scratch_cap >= got)
                rc = tsimd_tokenizer_decode_into(self->tok, ids, (size_t)n,
                                                 skip_special,
                                                 self->dec_scratch,
                                                 self->dec_scratch_cap, &got);
        }
        pthread_mutex_unlock(&self->mu);
        Py_END_ALLOW_THREADS
    }
    if (!have_view) free(ids);
    PyBuffer_Release(&view);
    if (rc != 0 || got == 0) {
        PyErr_SetString(TokError, "decode failed");
        return NULL;
    }
    return PyBytes_FromStringAndSize((const char *)self->dec_scratch,
                                     (Py_ssize_t)got);
}

static PyMethodDef Tok_methods[] = {
    {"encode", (PyCFunction)tok_encode, METH_VARARGS,
     "encode(bytes) -> bytes of uint32 LE ids (the fast path, auto threads)"},
    {"encode_list", (PyCFunction)tok_encode_list, METH_VARARGS,
     "encode_list(bytes) -> [ids] (auto threads)"},
    {"encode_bytes", (PyCFunction)tok_encode_bytes, METH_VARARGS,
     "encode_bytes(bytes) -> bytes of uint32 LE ids (auto threads)"},
    {"encode_threads", (PyCFunction)tok_encode_threads, METH_VARARGS,
     "encode_threads(bytes, threads=0) -> bytes of uint32 LE ids"},
    {"encode_threads_into", (PyCFunction)tok_encode_threads_into, METH_VARARGS,
     "encode_threads_into(bytes, bytearray_out, threads=0) -> id count"},
    {"encode_batch", (PyCFunction)tok_encode_batch, METH_VARARGS,
     "encode_batch(list_of_bytes, threads=0) -> [bytes of uint32 LE ids]"},
    {"decode_into", (PyCFunction)tok_decode_into, METH_VARARGS,
     "decode_into(ids_bytes, bytearray_out, skip_special=False, "
     "threads=0) -> the decoded byte count"},
    {"encode_bound", (PyCFunction)tok_encode_bound, METH_VARARGS,
     "encode_bound(len) -> required out-buffer bytes"},
    {"decode", (PyCFunction)tok_decode, METH_VARARGS,
     "decode(list_or_bytes_of_ids, skip_special=False) -> bytes"},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject TokType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tsimd.Tokenizer",
    .tp_basicsize = sizeof(TokObject),
    .tp_dealloc = (destructor)Tok_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_call = (ternaryfunc)tok_call,
    .tp_methods = Tok_methods,
};

static PyObject *mod_load(PyObject *self, PyObject *args) {
    const char *path;
    const char *format = NULL;
    const char *scheme = NULL;
    if (!PyArg_ParseTuple(args, "s|zz", &path, &format, &scheme)) return NULL;
    tsimd_model_format_t mt;
    PyObject *fmt_obj = format ? PyUnicode_FromString(format) : Py_None;
    if (!fmt_obj) return NULL;
    int frc = format_from_obj(fmt_obj, path, &mt);
    if (fmt_obj != Py_None) Py_DECREF(fmt_obj);
    if (frc != 0) return NULL;
    tsimd_pretok_scheme_t sch = TSIMD_PRETOK_GPT2;
    if (scheme) {
        if (strcmp(scheme, "o200k") == 0) sch = TSIMD_PRETOK_O200K;
        else if (strcmp(scheme, "cl100k") == 0) sch = TSIMD_PRETOK_CL100K;
        else if (strcmp(scheme, "kimi") == 0) sch = TSIMD_PRETOK_KIMI;
        else if (strcmp(scheme, "gpt2") != 0) {
            PyErr_Format(PyExc_ValueError, "unknown scheme: %s", scheme);
            return NULL;
        }
    }
    tsimd_tokenizer_t *tok = calloc(1, sizeof(tsimd_tokenizer_t));
    if (!tok) return PyErr_NoMemory();
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = tsimd_tokenizer_load_model(tok, mt, path, NULL, &sch, NULL, 0);
    Py_END_ALLOW_THREADS
    if (rc != 0) {
        free(tok);
        PyErr_SetString(TokError, "load failed");
        return NULL;
    }
    TokObject *obj = PyObject_New(TokObject, &TokType);
    if (!obj) {
        tsimd_tokenizer_free(tok);
        free(tok);
        return NULL;
    }
    obj->tok = tok;
    obj->scratch = NULL;
    obj->scratch_cap = 0;
    obj->dec_scratch = NULL;
    obj->dec_scratch_cap = 0;
    pthread_mutex_init(&obj->mu, NULL);
    return (PyObject *)obj;
}

static PyObject *mod_bytes_equal(PyObject *self, PyObject *args) {
    (void)self;
    Py_buffer a, b;
    if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    if (PyObject_GetBuffer((PyObject *)PyTuple_GetItem(args, 0), &a,
                           PyBUF_CONTIG_RO) != 0)
        return NULL;
    if (PyObject_GetBuffer((PyObject *)PyTuple_GetItem(args, 1), &b,
                           PyBUF_CONTIG_RO) != 0) {
        PyBuffer_Release(&a);
        return NULL;
    }
    int eq = a.len == b.len && (a.len == 0 ||
                                memcmp(a.buf, b.buf, (size_t)a.len) == 0);
    PyBuffer_Release(&a);
    PyBuffer_Release(&b);
    if (eq) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *mod_fold_doc(PyObject *self, PyObject *args) {
    unsigned long long h;
    Py_buffer view;
    if (!PyArg_ParseTuple(args, "Ky*", &h, &view)) return NULL;
    const uint8_t *p = (const uint8_t *)view.buf;
    size_t len = (size_t)view.len;
    size_t count = len / 4;
    static const uint64_t PRIME = 0x100000001b3ULL;
    for (int i = 0; i < 8; i++) {
        h ^= (uint8_t)(count >> (8 * i));
        h *= PRIME;
    }
    size_t head = count < 4096 ? count : 4096;
    size_t bytes = head * 4;
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= PRIME;
    }
    if (count > head) {
        size_t tail = (count - head) * 4;
        for (size_t i = 0; i < bytes; i++) {
            h ^= p[tail + i];
            h *= PRIME;
        }
    }
    PyBuffer_Release(&view);
    return PyLong_FromUnsignedLongLong(h);
}

static PyObject *mod_default_threads(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyLong_FromLong(tsimd_default_threads());
}

static PyMethodDef mod_methods[] = {
    {"load", mod_load, METH_VARARGS, "load(path, format=None, scheme=None) -> Tokenizer"},
    {"tokenizer", mod_load, METH_VARARGS, "tokenizer(path, format=None, scheme=None) -> Tokenizer"},
    {"default_threads", mod_default_threads, METH_NOARGS,
     "default_threads() -> the automatic worker count"},
    {"bytes_equal", mod_bytes_equal, METH_VARARGS,
     "bytes_equal(a, b) -> bool (the zero-copy memcmp)"},
    {"fold_doc", mod_fold_doc, METH_VARARGS,
     "fold_doc(state, ids_bytes) -> the next FNV-1a state folded over the "
     "document id count and the head and tail 4096 ids"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef mod_def = {
    PyModuleDef_HEAD_INIT,
    "tokensim",
    "tokensim tokenizer bindings",
    -1,
    mod_methods,
};

PyMODINIT_FUNC PyInit_tokensim(void) {
    if (PyType_Ready(&TokType) < 0) return NULL;
    PyObject *m = PyModule_Create(&mod_def);
    if (!m) return NULL;
    if (PyModule_AddStringConstant(m, "version", TSIMD_VERSION_STRING) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    TokError = PyErr_NewException("tokensim.Error", NULL, NULL);
    Py_XINCREF(TokError);
    if (PyModule_AddObject(m, "Error", TokError) < 0) {
        Py_XDECREF(TokError);
        Py_DECREF(m);
        return NULL;
    }
    Py_INCREF(&TokType);
    if (PyModule_AddObject(m, "Tokenizer", (PyObject *)&TokType) < 0) {
        Py_DECREF(&TokType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
