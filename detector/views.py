from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

import os
import json
import math
import pickle
import traceback
import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize, word_tokenize
from .utils.feature_extractor import calculate_all_features

# =========================
# CONFIG
# =========================
DEVICE = "cpu"
DEBERTA_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 256

# =========================
# PATHS
# =========================
ML_DIR = os.path.join(settings.BASE_DIR, "detector", "ml_assets")

CAT_PATH = os.path.join(ML_DIR, "catboost_numeric.cbm")
TEXT_LR_PATH = os.path.join(ML_DIR, "text_logreg.pkl")
STACKER_PATH = os.path.join(ML_DIR, "stacker_lgbm.pkl")
CALIBRATOR_PATH = os.path.join(ML_DIR, "calibrator.pkl")
SCALER_PATH = os.path.join(ML_DIR, "scaler.pkl")
FEATURE_ORDER_PATH = os.path.join(ML_DIR, "feature_order.pkl")
THRESHOLD_PATH = os.path.join(ML_DIR, "threshold.txt")

# =========================
# GLOBAL MODELS (LAZY)
# =========================
cat_model = None
text_lr = None
stacker = None
calibrator = None
scaler = None
feature_order = None
THRESHOLD = 0.5

deberta_tok = None
deberta = None
gpt2_tok = None
gpt2 = None
dgpt2_tok = None
dgpt2 = None

# =========================
# MODEL LOADERS
# =========================
def load_models():
    global cat_model, text_lr, stacker, calibrator, scaler, feature_order, THRESHOLD

    if cat_model is None:
        import catboost as cb

        cat_model = cb.CatBoostClassifier()
        cat_model.load_model(CAT_PATH)

        with open(TEXT_LR_PATH, "rb") as f:
            text_lr = pickle.load(f)
        with open(STACKER_PATH, "rb") as f:
            stacker = pickle.load(f)
        with open(CALIBRATOR_PATH, "rb") as f:
            calibrator = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(FEATURE_ORDER_PATH, "rb") as f:
            feature_order = pickle.load(f)
        with open(THRESHOLD_PATH) as f:
            THRESHOLD = float(f.read().strip())


def load_nlp_models():
    global deberta_tok, deberta, gpt2_tok, gpt2, dgpt2_tok, dgpt2

    if deberta is None:
        import torch
        from transformers import (
            DebertaV2Tokenizer,
            AutoModel,
            GPT2LMHeadModel,
            GPT2TokenizerFast
        )

        deberta_tok = DebertaV2Tokenizer.from_pretrained(DEBERTA_NAME)
        deberta = AutoModel.from_pretrained(DEBERTA_NAME).eval()

        gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")
        gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").eval()

        dgpt2_tok = GPT2TokenizerFast.from_pretrained("distilgpt2")
        dgpt2 = GPT2LMHeadModel.from_pretrained("distilgpt2").eval()

# =========================
# NLP HELPERS
# =========================
def perplexity(text, model, tokenizer):
    import torch
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return math.exp(loss.item())


def runtime_nlp_features(text):
    import torch
    load_nlp_models()

    p1 = perplexity(text, gpt2, gpt2_tok)
    p2 = perplexity(text, dgpt2, dgpt2_tok)

    ents = []
    for s in sent_tokenize(text):
        toks = word_tokenize(s.lower())
        if len(toks) > 1:
            p = pd.Series(toks).value_counts(normalize=True).values
            ents.append(-np.sum(p * np.log(p + 1e-9)))

    ent_mean, ent_std = (np.mean(ents), np.std(ents)) if ents else (0.0, 0.0)

    enc = gpt2_tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = gpt2(**enc).logits[0]

    ranks = []
    ids = enc["input_ids"][0]
    for i in range(len(ids) - 1):
        probs = torch.softmax(logits[i], dim=-1)
        ranks.append((probs > probs[ids[i + 1]]).sum().item())

    rank_ent = 0.0
    if ranks:
        p = np.array(ranks) / (np.sum(ranks) + 1e-9)
        rank_ent = -np.sum(p * np.log(p + 1e-9))

    return np.array([[p1, p2, p1 - p2, rank_ent]], dtype=np.float32)


def deberta_embedding(text):
    import torch
    load_nlp_models()

    enc = deberta_tok([text], padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        out = deberta(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    pooled = (out * mask).sum(1) / mask.sum(1)
    return pooled.cpu().numpy()

# =========================
# VIEWS
# =========================
def home(request):
    return render(request, "index.html")


def detect_text(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        load_models()

        data = json.loads(request.body)
        text = data.get("text_input", "").strip()

        feats = calculate_all_features(text)
        X_ling = np.array([[feats.get(f, 0.0) for f in feature_order]], dtype=np.float32)
        X_ling = scaler.transform(X_ling)

        X_runtime = runtime_nlp_features(text)
        X_num = np.hstack([X_ling, X_runtime])
        p_num = cat_model.predict_proba(X_num)[:, 1]

        emb = deberta_embedding(text)
        p_text = text_lr.predict_proba(emb)[:, 1]

        X_stack = np.column_stack([p_num, p_text])
        p_raw = stacker.predict_proba(X_stack)[:, 1]
        p_final = calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]

        pred = "ai" if p_final[0] >= THRESHOLD else "human"

        return JsonResponse({
            "prediction": pred,
            "ai_probability": round(float(p_final[0]), 4)
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)
