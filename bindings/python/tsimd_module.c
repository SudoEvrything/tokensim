#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "tokenizer.h"
#include "tokensimd_version.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    PyObject_HEAD
    tsimd_tokenizer_t *tok;
} TokObject;

static PyObject *TokError;

static void Tok_dealloc(TokObject *self) {
    if (self->tok) {
        tsimd_tokenizer_free(self->tok);
        free(self->tok);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int format_from_obj(PyObject *obj, tsimd_model_format_t *out) {
    if (obj == NULL || obj == Py_None) {
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
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = tsimd_tokenizer_encode(self->tok,
                                (const uint8_t *)view.buf,
                                (size_t)view.len, &seq);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (rc != 0) {
        PyErr_SetString(TokError, "encode failed");
        return NULL;
    }
    PyObject *list = PyList_New((Py_ssize_t)seq.count);
    if (!list) {
        tsimd_token_sequence_free(&seq);
        return NULL;
    }
    for (size_t i = 0; i < seq.count; i++) {
        PyObject *v = PyLong_FromUnsignedLong(seq.ids[i]);
        if (!v || PyList_SetItem(list, (Py_ssize_t)i, v) != 0) {
            Py_XDECREF(v);
            Py_DECREF(list);
            tsimd_token_sequence_free(&seq);
            return NULL;
        }
    }
    tsimd_token_sequence_free(&seq);
    return list;
}

static PyObject *tok_encode_bytes(TokObject *self, PyObject *args) {
    Py_buffer view;
    if (!PyArg_ParseTuple(args, "y*", &view)) return NULL;
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = tsimd_tokenizer_encode(self->tok,
                                (const uint8_t *)view.buf,
                                (size_t)view.len, &seq);
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

static PyObject *tok_encode_threads(TokObject *self, PyObject *args) {
    Py_buffer view;
    int threads = 6;
    if (!PyArg_ParseTuple(args, "y*|i", &view, &threads)) return NULL;
    tsimd_token_sequence_t seq;
    memset(&seq, 0, sizeof seq);
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = tsimd_tokenizer_encode_parallel(self->tok,
                                         (const uint8_t *)view.buf,
                                         (size_t)view.len, threads, &seq);
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
    int threads = 6;
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
    rc = tsimd_tokenizer_encode_parallel_into(self->tok,
                                              (const uint8_t *)view.buf,
                                              (size_t)view.len, threads,
                                              (uint32_t *)ov.buf, cap,
                                              &count);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&ov);
    PyBuffer_Release(&view);
    if (rc != 0) {
        PyErr_SetString(TokError, "encode failed");
        return NULL;
    }
    return PyLong_FromSize_t(count);
}

static PyObject *tok_decode(TokObject *self, PyObject *args) {
    PyObject *list;
    int skip_special = 0;
    if (!PyArg_ParseTuple(args, "O!|p", &PyList_Type, &list, &skip_special))
        return NULL;
    Py_ssize_t n = PyList_Size(list);
    uint32_t *ids = malloc((size_t)(n > 0 ? n : 1) * sizeof(uint32_t));
    if (!ids) return PyErr_NoMemory();
    for (Py_ssize_t i = 0; i < n; i++) {
        unsigned long v = PyLong_AsUnsignedLong(PyList_GetItem(list, i));
        if (v == (unsigned long)-1 && PyErr_Occurred()) {
            free(ids);
            return NULL;
        }
        ids[i] = (uint32_t)v;
    }
    tsimd_byte_sequence_t out;
    memset(&out, 0, sizeof out);
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = tsimd_tokenizer_decode(self->tok, ids, (size_t)n, skip_special,
                                &out);
    Py_END_ALLOW_THREADS
    free(ids);
    if (rc != 0) {
        PyErr_SetString(TokError, "decode failed");
        return NULL;
    }
    PyObject *res = PyBytes_FromStringAndSize((const char *)out.bytes,
                                              (Py_ssize_t)out.len);
    tsimd_byte_sequence_free(&out);
    return res;
}

static PyMethodDef Tok_methods[] = {
    {"encode", (PyCFunction)tok_encode, METH_VARARGS,
     "encode(bytes) -> [ids]"},
    {"encode_bytes", (PyCFunction)tok_encode_bytes, METH_VARARGS,
     "encode_bytes(bytes) -> bytes of uint32 LE ids"},
    {"encode_threads", (PyCFunction)tok_encode_threads, METH_VARARGS,
     "encode_threads(bytes, threads=6) -> bytes of uint32 LE ids"},
    {"encode_threads_into", (PyCFunction)tok_encode_threads_into, METH_VARARGS,
     "encode_threads_into(bytes, bytearray_out, threads=6) -> id count"},
    {"encode_bound", (PyCFunction)tok_encode_bound, METH_VARARGS,
     "encode_bound(len) -> required out-buffer bytes"},
    {"decode", (PyCFunction)tok_decode, METH_VARARGS,
     "decode([ids], skip_special=False) -> bytes"},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject TokType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tsimd.Tokenizer",
    .tp_basicsize = sizeof(TokObject),
    .tp_dealloc = (destructor)Tok_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = Tok_methods,
};

static PyObject *mod_load(PyObject *self, PyObject *args) {
    const char *path;
    const char *format = NULL;
    const char *scheme = NULL;
    if (!PyArg_ParseTuple(args, "s|zz", &path, &format, &scheme)) return NULL;
    tsimd_model_format_t mt;
    if (format_from_obj(format ? PyUnicode_FromString(format) : Py_None,
                        &mt) != 0)
        return NULL;
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
    return (PyObject *)obj;
}

static PyMethodDef mod_methods[] = {
    {"load", mod_load, METH_VARARGS, "load(path, format=None, scheme=None) -> Tokenizer"},
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
