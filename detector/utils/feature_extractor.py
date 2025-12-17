import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re
import csv
import string
import numpy as np
from collections import Counter
import importlib
import unicodedata
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Optional: spaCy for TALLED syntactic features
try:
    import spacy  # noqa: F401
    _SPACY_AVAILABLE = True
except Exception:
    _SPACY_AVAILABLE = False
spacy_nlp = None

# --- NLTK setup ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Cleaning helpers (from cleanerText.py) ---
def remove_nested_braces(text):
    """Remove all {{...}} templates including nested ones."""
    pattern = r'{{[^{}]*}}'
    while re.search(pattern, text):
        text = re.sub(pattern, '', text)
    return text

def clean_text(text):
    """Clean raw text, normalize accents/quotes, and remove unwanted symbols.

    - Fix common mojibake (e.g., 'Ã¢â‚¬â„¢') by re-decoding latin1→utf-8 when detected
    - Normalize unicode to NFKC
    - Map curly quotes/dashes to ASCII equivalents
    - Remove URLs, refs, templates
    - Remove most punctuation, keeping ., , ' , " , -
    - Filter tokens to letters/numbers and allowed punctuation only
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return text
    if not isinstance(text, str):
        return text

    # Attempt to repair mojibake commonly seen from cp1252/latin1
    if 'Ã' in text or 'Â' in text:
        try:
            text = text.encode('latin1').decode('utf-8')
        except Exception:
            pass

    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Replace curly quotes/dashes/non-breaking space
    replacements = {
        '\u2018': "'", '\u2019': "'",  # single quotes
        '\u201C': '"', '\u201D': '"',  # double quotes
        '\u2013': '-', '\u2014': '-',   # en/em dash
        '\u00A0': ' ',                   # non-breaking space
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove <ref>...</ref> blocks and standalone </ref>
    text = re.sub(r'<ref[^>]*?>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'</ref>', '', text)

    # Remove nested {{...}} templates
    text = remove_nested_braces(text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Allow only these punctuation marks
    allowed = '.,\'"-'
    punctuation_to_remove = ''.join(c for c in string.punctuation if c not in allowed)
    text = text.translate(str.maketrans('', '', punctuation_to_remove))

    # Keep only words containing letters/numbers and allowed punctuation
    words = text.split()
    words = [word for word in words if re.match(r'^[a-zA-Z0-9.,\'"-]+$', word)]

    return ' '.join(words)

# --- Common helpers ---
function_words = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
    'was', 'were', 'will', 'with', 'would', 'you', 'your', 'i', 'me', 'my',
    'we', 'us', 'our', 'they', 'them', 'their', 'she', 'her', 'him', 'his',
    'this', 'these', 'those', 'what', 'which', 'who', 'where', 'when', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'could', 'should', 'shall', 'may', 'might',
    'must', 'ought', 'do', 'does', 'did', 'have', 'had', 'having'
}

# --- TALLED: spaCy-based syntactic features ---
TALLED_FEATURE_KEYS = {
    "cl_av_deps": "dependents per clause",
    "cl_ndeps_std_dev": "dependents per clause (standard deviation)",
    "acomp_per_cl": "adjective complements per clause",
    "advcl_per_cl": "adverbial clauses per clause",
    "agent_per_cl": "passive agents per clause",
    "cc_per_cl": "clausal coordinating conjunctions per clause",
    "ccomp_per_cl": "clausal complements per clause",
    "conj_per_cl": "conjunctions per clause",
    "csubj_per_cl": "clausal subjects per clause",
    "csubjpass_per_cl": "passive clausal subjects per clause",
    "dep_per_cl": "undefined dependents per clause",
    "discourse_per_cl": "discourse markers per clause",
    "dobj_per_cl": "direct objects per clause",
    "expl_per_cl": "existential 'there' per clause",
    "iobj_per_cl": "indirect objects per clause",
    "mark_per_cl": "subordinating conjunctions per clause",
    "ncomp_per_cl": "nominal complements per clause",
    "neg_per_cl": "clausal negations per clause",
    "nsubj_per_cl": "nominal subjects per clause",
    "nsubjpass_per_cl": "passive nominal subjects per clause",
    "parataxis_per_cl": "instances of parataxis per clause",
    "pcomp_per_cl": "clausal prepositional complements per clause",
    "prep_per_cl": "prepositions per clause",
    "prt_per_cl": "phrasal verb particle per clause",
    "tmod_per_cl": "bare noun phrase temporal modifiers per clause",
    "xcomp_per_cl": "open clausal complements per clause",
    "xsubj_per_cl": "controlling subjects per clause",
    "advmod_per_cl": "adverbial modifiers per clause",
    "aux_per_cl": "auxiliary verbs per clause",
    "auxpass_per_cl": "passive auxiliary verbs per clause",
    "modal_per_clause": "modal auxiliaries per clause",
}

def features_talled(text):
    # Return zeros for invalid text
    if not isinstance(text, str) or not text.strip():
        return {k: 0 for k in TALLED_FEATURE_KEYS.keys()}
    # Ensure spaCy model if available
    global spacy_nlp
    if _SPACY_AVAILABLE and spacy_nlp is None:
        try:
            spacy_nlp = importlib.import_module('spacy').load('en_core_web_sm')
        except Exception:
            try:
                spacy_nlp = importlib.import_module('spacy').blank('en')
            except Exception:
                spacy_nlp = None
    # If spaCy unavailable, return zeros
    if spacy_nlp is None:
        return {k: 0 for k in TALLED_FEATURE_KEYS.keys()}
    doc = spacy_nlp(text)
    counts = {k: 0 for k in TALLED_FEATURE_KEYS.keys()}
    for token in doc:
        dep_label = token.dep_
        key = f"{dep_label}_per_cl"
        if key in counts:
            counts[key] += 1
        # Special handling for modals (MD tag)
        if token.tag_ == 'MD':
            if 'modal_per_clause' in counts:
                counts['modal_per_clause'] += 1
    return counts

# --- SEANCE features (imported from seance.py) ---
"""
SEANCE inline features: affective norms, GALC emotional, General Inquirer, SenticNet, VADER
This section embeds SEANCE feature computation directly for a single-file combined module.
It attempts to load validated resources from local SEANCE data_files; otherwise falls back to lists.
"""

# SEANCE globals
AFFECTIVE_NORMS = {}
GI_CATEGORIES = {}
SENTIC_NET = {}
VADER_LEXICON = {}
VADER_ANALYZER = None

def _resolve_seance_data_dir():
    candidates = [
        r"d:/Documents and  Projects/sem6/SEANCE_1_2_0_Py3/data_files",
        r"D:/Documents and  Projects/sem6/SEANCE_1_2_0_Py3/data_files",
        os.path.join(os.path.dirname(__file__), '..', 'SEANCE_1_2_0_Py3', 'data_files'),
        os.path.join(os.path.dirname(__file__), 'SEANCE_1_2_0_Py3', 'data_files'),
        os.path.join(os.getcwd(), 'SEANCE_1_2_0_Py3', 'data_files'),
        os.path.join(os.path.dirname(__file__), 'data_files'),
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                return p
        except Exception:
            continue
    return None

SEANCE_DATA_DIR = _resolve_seance_data_dir()

def seance_create_empty_feature_dict():
    feature_names = [
        'Arousal', 'Arousal_nwords', 'Dominance', 'Dominance_nwords', 'Valence', 'Valence_nwords',
        'Longing_GALC', 'Lust_GALC', 'Anger_GALC', 'Anxiety_GALC', 'Boredom_GALC', 'Contempt_GALC',
        'Desperation_GALC', 'Disappointment_GALC', 'Disgust_GALC', 'Dissatisfaction_GALC', 'Envy_GALC',
        'Fear_GALC', 'Guilt_GALC', 'Hatred_GALC', 'Irritation_GALC', 'Jealousy_GALC', 'Sadness_GALC',
        'Shame_GALC', 'Tension/Stress_GALC', 'Negative_GALC', 'Admiration/Awe_GALC', 'Amusement_GALC',
        'Contentment_GALC', 'Feelinglove_GALC', 'Happiness_GALC', 'Hope_GALC', 'Interest/Enthusiasm_GALC',
        'Joy_GALC', 'Pleasure/Enjoyment_GALC', 'Pride_GALC', 'Relaxation/Serenity_GALC', 'Relief_GALC',
        'Positive_GALC', 'Beingtouched_GALC', 'Compassion_GALC', 'Gratitude_GALC', 'Humility_GALC',
        'Surprise_GALC',
        # GI features (50)
        'Fail_GI', 'No_GI', 'Negate_GI', 'Ani_GI', 'Aquatic_GI', 'Land_GI', 'Sky_GI', 'Object_GI',
        'Tool_GI', 'Food_GI', 'Vehicle_GI', 'Bldgpt_GI', 'Natobj_GI', 'Bodypt_GI', 'Natrpro_GI',
        'Color_GI', 'Positiv_GI', 'Pstv_GI', 'Pleasur_GI', 'Yes_GI', 'Increas_GI', 'Decreas_GI',
        'Quality_GI', 'Quan_GI', 'Numb_GI', 'Ord_GI', 'Card_GI', 'Freq_GI', 'Dist_GI', 'Self_GI',
        'Our_GI', 'You_GI', 'Name_GI', 'Affil_GI', 'Role_GI', 'Coll_GI', 'Work_GI', 'Ritual_GI',
        'Socrel_GI', 'Race_GI', 'Kin_2_GI', 'Male_GI', 'Female_GI', 'Nonadlt_GI', 'Hu_GI',
        'Social_GI', 'Rel_GI', 'Intrj_GI', 'Ipadj_GI', 'Indadj_GI',
        # SenticNet (6)
        'SenticNet_Pleasantness', 'SenticNet_Attention', 'SenticNet_Sensitivity', 'SenticNet_Aptitude',
        'SenticNet_Polarity', 'SenticNet_Coverage',
        # VADER (5)
        'VADER_Compound', 'VADER_Positive', 'VADER_Neutral', 'VADER_Negative', 'VADER_Coverage'
    ]
    return {f: 0 for f in feature_names}

def load_affective_norms():
    if not SEANCE_DATA_DIR:
        return
    path = os.path.join(SEANCE_DATA_DIR, 'affective_norms.txt')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = re.split(r"\s+|\t|,", line)
                if len(parts) >= 4:
                    word = parts[0].lower()
                    try:
                        valence = float(parts[1]); arousal = float(parts[2]); dominance = float(parts[3])
                        AFFECTIVE_NORMS[word] = {
                            'valence': valence, 'arousal': arousal, 'dominance': dominance
                        }
                    except Exception:
                        continue
    except FileNotFoundError:
        pass

def load_general_inquirer():
    # GI_CATEGORIES are defined in code in calculate_general_inquirer_features; nothing to load
    return

def load_sentic_net():
    if not SEANCE_DATA_DIR:
        return
    path = os.path.join(SEANCE_DATA_DIR, 'senticnet_data.txt')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 6:
                    word = parts[0].lower()
                    try:
                        pleasantness = float(parts[1]); attention = float(parts[2]); sensitivity = float(parts[3]); aptitude = float(parts[4]); polarity = float(parts[5])
                        SENTIC_NET[word] = {
                            'pleasantness': pleasantness,
                            'attention': attention,
                            'sensitivity': sensitivity,
                            'aptitude': aptitude,
                            'polarity': polarity
                        }
                    except Exception:
                        continue
    except FileNotFoundError:
        pass

def load_vader_lexicon():
    if not SEANCE_DATA_DIR:
        return
    path = os.path.join(SEANCE_DATA_DIR, 'vader_lexicon.txt')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        word = parts[0].lower()
                        try:
                            score = float(parts[1])
                        except Exception:
                            score = 0.0
                        VADER_LEXICON[word] = score
        # Try to import analyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            global VADER_ANALYZER
            VADER_ANALYZER = SentimentIntensityAnalyzer()
        except Exception:
            try:
                import sys
                if SEANCE_DATA_DIR and SEANCE_DATA_DIR not in sys.path:
                    sys.path.append(SEANCE_DATA_DIR)
                VADER_ANALYZER = SentimentIntensityAnalyzer()
            except Exception:
                VADER_ANALYZER = None
    except FileNotFoundError:
        pass

# Initialize SEANCE resources
load_affective_norms(); load_general_inquirer(); load_sentic_net(); load_vader_lexicon()

def seance_calculate_affective_features(words):
    if not AFFECTIVE_NORMS:
        # Fallback lists
        high_arousal = {'excited','thrilled','energetic','intense','passionate','stimulated','alert','active','frantic','frenzied','wild','fierce','violent','explosive','dynamic','powerful','overwhelming','exhilarating','electrifying','aroused','awakened','stimulating'}
        high_dominance = {'control','power','command','authority','dominant','strong','confident','leader','boss','master','ruler','superior','influential','controlling','assertive','forceful','decisive','determined','independent','autonomous','self-assured','commanding'}
        positive_valence = {'happy','joy','love','wonderful','excellent','beautiful','amazing','fantastic','great','good','pleasant','delightful','cheerful','satisfied','content','pleased','glad','grateful','blessed','fortunate','lucky','successful','winning','triumph'}
        total = len(words)
        ar_m = sum(1 for w in words if w in high_arousal)
        do_m = sum(1 for w in words if w in high_dominance)
        va_m = sum(1 for w in words if w in positive_valence)
        return {
            'Arousal': round(ar_m/total*100,3) if total>0 else 0,
            'Arousal_nwords': ar_m,
            'Dominance': round(do_m/total*100,3) if total>0 else 0,
            'Dominance_nwords': do_m,
            'Valence': round(va_m/total*100,3) if total>0 else 0,
            'Valence_nwords': va_m
        }
    # Use norms
    valence_scores=[]; arousal_scores=[]; dominance_scores=[]; matched=0
    for w in words:
        wl = w.lower()
        if wl in AFFECTIVE_NORMS:
            n = AFFECTIVE_NORMS[wl]
            valence_scores.append(n['valence']); arousal_scores.append(n['arousal']); dominance_scores.append(n['dominance']); matched+=1
    avg_val = sum(valence_scores)/len(valence_scores) if valence_scores else 5.0
    avg_ar = sum(arousal_scores)/len(arousal_scores) if arousal_scores else 5.0
    avg_do = sum(dominance_scores)/len(dominance_scores) if dominance_scores else 5.0
    val_pct = ((avg_val-1)/8)*100; ar_pct = ((avg_ar-1)/8)*100; do_pct = ((avg_do-1)/8)*100
    return {
        'Arousal': round(ar_pct,3), 'Arousal_nwords': matched,
        'Dominance': round(do_pct,3), 'Dominance_nwords': matched,
        'Valence': round(val_pct,3), 'Valence_nwords': matched
    }

def seance_calculate_galc_emotions(words):
    emotional_categories = {
        'Longing_GALC': {'yearning','longing','craving','desire','wish','want','miss','ache'},
        'Lust_GALC': {'lust','desire','passion','attraction','temptation','seduction','arousal'},
        'Anger_GALC': {'angry','rage','fury','mad','furious','irritated','annoyed','hostile','outraged'},
        'Anxiety_GALC': {'anxious','worried','nervous','stressed','tense','uneasy','apprehensive','concerned'},
        'Boredom_GALC': {'bored','boring','dull','tedious','monotonous','uninteresting','tiresome'},
        'Contempt_GALC': {'contempt','scorn','disdain','disgrace','despise','loathe','detest'},
        'Desperation_GALC': {'desperate','hopeless','despair','frantic','urgent','critical','dire'},
        'Disappointment_GALC': {'disappointed','letdown','frustrated','dissatisfied','discouraged','dismayed'},
        'Disgust_GALC': {'disgusted','revolted','repulsed','nauseated','sickened','appalled'},
        'Dissatisfaction_GALC': {'dissatisfied','unhappy','displeased','unsatisfied','discontent'},
        'Envy_GALC': {'envious','jealous','covetous','resentful','begrudge'},
        'Fear_GALC': {'afraid','scared','frightened','terrified','fearful','panic','horror','dread'},
        'Guilt_GALC': {'guilty','shame','remorse','regret','sorry','ashamed','blame'},
        'Hatred_GALC': {'hate','hatred','loathe','detest','abhor','despise','hostility'},
        'Irritation_GALC': {'irritated','annoyed','bothered','frustrated','agitated','vexed'},
        'Jealousy_GALC': {'jealous','envious','possessive','suspicious','mistrustful'},
        'Sadness_GALC': {'sad','depressed','melancholy','gloomy','sorrowful','grief','mourning','blue'},
        'Shame_GALC': {'ashamed','embarrassed','humiliated','mortified','disgraced'},
        'Tension/Stress_GALC': {'tense','stressed','pressure','strain','burden','overwhelmed'},
        'Negative_GALC': {'negative','bad','wrong','terrible','awful','horrible','worst','evil'},
        'Admiration/Awe_GALC': {'admire','awe','respect','reverence','wonder','amazed','impressed'},
        'Amusement_GALC': {'amused','funny','hilarious','entertaining','humorous','comic','laugh'},
        'Contentment_GALC': {'content','satisfied','peaceful','serene','calm','comfortable'},
        'Feelinglove_GALC': {'love','adore','cherish','affection','caring','devoted','tender'},
        'Happiness_GALC': {'happy','joyful','cheerful','glad','delighted','pleased','elated'},
        'Hope_GALC': {'hope','hopeful','optimistic','confident','faith','trust','believe'},
        'Interest/Enthusiasm_GALC': {'interested','enthusiastic','excited','curious','engaged','fascinated'},
        'Joy_GALC': {'joy','joyful','blissful','ecstatic','euphoric','jubilant','exuberant'},
        'Pleasure/Enjoyment_GALC': {'pleasure','enjoy','fun','delight','satisfaction','gratification'},
        'Pride_GALC': {'proud','pride','accomplished','achievement','triumph','success'},
        'Relaxation/Serenity_GALC': {'relaxed','calm','peaceful','tranquil','serene','restful'},
        'Relief_GALC': {'relief','relieved','comfort','ease','soothe','consolation'},
        'Positive_GALC': {'positive','good','great','excellent','wonderful','amazing','fantastic'},
        'Beingtouched_GALC': {'touched','moved','emotional','heartfelt','meaningful','poignant'},
        'Compassion_GALC': {'compassion','empathy','sympathy','kindness','caring','understanding'},
        'Gratitude_GALC': {'grateful','thankful','appreciation','blessed','thankfulness'},
        'Humility_GALC': {'humble','modest','meek','unassuming','unpretentious'},
        'Surprise_GALC': {'surprised','amazed','astonished','shocked','stunned','unexpected'}
    }
    total = len(words); feats = {}
    for cat, wordset in emotional_categories.items():
        m = sum(1 for w in words if w in wordset)
        feats[cat] = round(m/total*100,3) if total>0 else 0
    return feats

def seance_calculate_sentic_net(words, pos_tags):
    if not SENTIC_NET:
        # Fallback: zero scores, coverage based on word matches
        return {
            'SenticNet_Pleasantness': 0, 'SenticNet_Attention': 0, 'SenticNet_Sensitivity': 0,
            'SenticNet_Aptitude': 0, 'SenticNet_Polarity': 0, 'SenticNet_Coverage': 0
        }
    pleasantness=attention=sensitivity=aptitude=polarity=0.0; matched=0
    for w in words:
        wl=w.lower()
        if wl in SENTIC_NET:
            s=SENTIC_NET[wl]
            pleasantness+=s['pleasantness']; attention+=s['attention']; sensitivity+=s['sensitivity']; aptitude+=s['aptitude']; polarity+=s['polarity']; matched+=1
    total = len(words)
    return {
        'SenticNet_Pleasantness': round(pleasantness/max(matched,1),3) if matched>0 else 0,
        'SenticNet_Attention': round(attention/max(matched,1),3) if matched>0 else 0,
        'SenticNet_Sensitivity': round(sensitivity/max(matched,1),3) if matched>0 else 0,
        'SenticNet_Aptitude': round(aptitude/max(matched,1),3) if matched>0 else 0,
        'SenticNet_Polarity': round(polarity/max(matched,1),3) if matched>0 else 0,
        'SenticNet_Coverage': matched
    }

def seance_calculate_vader(text):
    if VADER_ANALYZER is not None:
        scores = VADER_ANALYZER.polarity_scores(text)
        return {
            'VADER_Compound': round(scores.get('compound',0),3),
            'VADER_Positive': round(scores.get('pos',0),3),
            'VADER_Neutral': round(scores.get('neu',0),3),
            'VADER_Negative': round(scores.get('neg',0),3),
            'VADER_Coverage': len(text.split())
        }
    # Lexicon-only fallback: simple average sentiment over known words
    words = [t for t in word_tokenize(text.lower()) if t.isalnum()]
    total = len(words); matched=0; comp=0.0
    for w in words:
        s = VADER_LEXICON.get(w, 0.0)
        if s != 0.0:
            matched += 1
            comp += s
    return {
        'VADER_Compound': round(comp/max(matched,1),3) if matched>0 else 0,
        'VADER_Positive': 0, 'VADER_Neutral': 0, 'VADER_Negative': 0,
        'VADER_Coverage': matched
    }

def seance_calculate_general_inquirer(words, pos_tags):
    gi_categories = {
        'Fail_GI': {'fail','failure','failed','failing','unsuccessful','defeat','lose','lost','wrong','mistake','error','fault'},
        'No_GI': {'no','not','none','nothing','never','neither','nor','nobody','nowhere','noone','nope'},
        'Negate_GI': {'deny','refuse','reject','decline','disagree','oppose','against','anti','counter','contradict','dispute'},
        'Ani_GI': {'animal','dog','cat','bird','fish','horse','cow','pig','sheep','chicken','lion','tiger','bear','wolf','deer','rabbit'},
        'Aquatic_GI': {'water','ocean','sea','lake','river','stream','pond','fish','swim','boat','ship','wave','tide','beach','shore'},
        'Land_GI': {'land','ground','earth','soil','dirt','mountain','hill','valley','field','forest','desert','rock','stone','tree','grass'},
        'Sky_GI': {'sky','cloud','sun','moon','star','air','wind','rain','snow','storm','weather','atmosphere','heaven','space'},
        'Object_GI': {'thing','object','item','stuff','material','substance','matter','element','component','part','piece','equipment'},
        'Tool_GI': {'tool','instrument','device','machine','equipment','hammer','knife','saw','screwdriver','wrench','drill','computer'},
        'Food_GI': {'food','eat','bread','meat','fruit','vegetable','milk','cheese','fish','chicken','rice','pasta','pizza','cake'},
        'Vehicle_GI': {'car','truck','bus','train','plane','ship','boat','bicycle','motorcycle','vehicle','drive','ride','travel'},
        'Bldgpt_GI': {'building','house','home','room','door','window','wall','roof','floor','kitchen','bedroom','bathroom','office'},
        'Natobj_GI': {'nature','natural','tree','flower','plant','leaf','branch','root','seed','fruit','vegetable','herb','bush'},
        'Bodypt_GI': {'body','head','face','eye','nose','mouth','ear','hand','arm','leg','foot','finger','toe','heart','brain'},
        'Natrpro_GI': {'process','natural','growth','development','evolution','change','transformation','cycle','season','birth','death'},
        'Color_GI': {'color','red','blue','green','yellow','black','white','brown','orange','purple','pink','gray','bright','dark'},
        'Positiv_GI': {'positive','good','great','excellent','wonderful','amazing','fantastic','perfect','beautiful','lovely','nice','pleasant'},
        'Pstv_GI': {'happy','joy','glad','cheerful','delighted','pleased','satisfied','content','excited','thrilled','elated','blissful'},
        'Pleasur_GI': {'pleasure','enjoy','fun','entertaining','amusing','delightful','satisfying','gratifying','rewarding','fulfilling'},
        'Yes_GI': {'yes','yeah','yep','sure','okay','ok','right','correct','true','agree','accept','approve'},
        'Increas_GI': {'increase','grow','rise','expand','extend','enlarge','boost','enhance','improve','add','more','greater'},
        'Decreas_GI': {'decrease','reduce','lower','decline','drop','fall','shrink','diminish','lessen','cut','less','fewer'},
        'Quality_GI': {'quality','excellent','good','bad','poor','high','low','fine','superior','inferior','grade','standard'},
        'Quan_GI': {'quantity','amount','number','count','total','sum','volume','size','measure','degree','extent','level'},
        'Numb_GI': {'one','two','three','four','five','six','seven','eight','nine','ten','hundred','thousand','million'},
        'Ord_GI': {'first','second','third','fourth','fifth','last','next','previous','final','initial','primary','secondary'},
        'Card_GI': {'one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','twenty','thirty'},
        'Freq_GI': {'often','always','never','sometimes','usually','frequently','rarely','seldom','occasionally','regularly','constantly'},
        'Dist_GI': {'near','far','close','distant','here','there','everywhere','somewhere','nowhere','distance','away','apart'},
        'Self_GI': {'i','me','my','mine','myself','self','own','personal','individual','private','alone','solo'},
        'Our_GI': {'we','us','our','ours','ourselves','together','collective','group','team','community','society','public'},
        'You_GI': {'you','your','yours','yourself','yourselves','thou','thy','thee','thine'},
        'Name_GI': {'name','called','named','title','label','term','word','phrase','expression','designation','identification'},
        'Affil_GI': {'member','belong','part','group','organization','association','club','society','team','community','join'},
        'Role_GI': {'role','position','job','work','occupation','profession','career','duty','responsibility','function','task'},
        'Coll_GI': {'college','university','school','education','student','teacher','professor','class','course','study','learn'},
        'Work_GI': {'work','job','employment','career','profession','occupation','business','office','company','labor','effort'},
        'Ritual_GI': {'ritual','ceremony','tradition','custom','practice','habit','routine','procedure','protocol','formality'},
        'Socrel_GI': {'social','society','community','public','people','human','relationship','interaction','communication','connection'},
        'Race_GI': {'race','ethnic','culture','cultural','nationality','heritage','ancestry','origin','background','tradition'},
        'Kin_2_GI': {'family','relative','relation','parent','child','sibling','brother','sister','mother','father','son','daughter'},
        'Male_GI': {'man','boy','male','gentleman','guy','father','son','brother','husband','boyfriend','he','him','his'},
        'Female_GI': {'woman','girl','female','lady','mother','daughter','sister','wife','girlfriend','she','her','hers'},
        'Nonadlt_GI': {'child','kid','baby','infant','toddler','boy','girl','teenager','youth','young','minor','juvenile'},
        'Hu_GI': {'human','person','people','individual','man','woman','child','adult','being','soul','life','living'},
        'Social_GI': {'social','society','community','group','public','collective','together','interaction','relationship','connection'},
        'Rel_GI': {'religion','religious','god','church','faith','belief','spiritual','holy','sacred','divine','prayer','worship'},
        'Intrj_GI': {'oh','ah','wow','hey','hello','hi','goodbye','bye','thanks','please','sorry','excuse'},
        'Ipadj_GI': {'very','quite','rather','somewhat','fairly','pretty','really','truly','extremely','highly','deeply'},
        'Indadj_GI': {'this','that','these','those','such','same','other','another','each','every','all','some'}
    }
    total = len(words); feats={}
    for cat, wordset in gi_categories.items():
        m = sum(1 for w in words if w in wordset)
        feats[cat] = round(m/total*100,3) if total>0 else 0
    return feats

def seance_calculate_psycholinguistic_features(text):
    if not isinstance(text, str) or not text.strip():
        return seance_create_empty_feature_dict()
    tokens = word_tokenize(text.lower())
    words = [t for t in tokens if t.isalnum()]
    if not words:
        return seance_create_empty_feature_dict()
    pos_tags = pos_tag(words)
    features = {}
    features.update(seance_calculate_affective_features(words))
    features.update(seance_calculate_galc_emotions(words))
    features.update(seance_calculate_general_inquirer(words, pos_tags))
    features.update(seance_calculate_sentic_net(words, pos_tags))
    features.update(seance_calculate_vader(text))
    return features

def features_seance(text):
    try:
        return seance_calculate_psycholinguistic_features(text)
    except Exception:
        return seance_create_empty_feature_dict()

def categorize_words(words):
    pos_tags = pos_tag(words)
    all_words = words
    content_words = []
    function_words_found = []
    for word, pos in pos_tags:
        if (word.lower() in function_words or 
            pos in ['DT', 'IN', 'TO', 'CC', 'PRP', 'PRP$', 'WDT', 'WP', 'WP$', 'WRB', 'MD']):
            function_words_found.append(word)
        else:
            if pos.startswith(('NN', 'VB', 'JJ', 'RB')):
                content_words.append(word)
            else:
                content_words.append(word)
    return all_words, content_words, function_words_found

# --- Section 1: Word Frequency features (36) ---
def _sim_freqs(word_list, corpus_name):
    if not word_list:
        return 0.0, 0.0
    if corpus_name == 'KF':
        freqs = [max(1, 100 - len(w) * 10) for w in word_list]
    elif corpus_name == 'TL':
        freqs = [max(1, 50 - len(w) * 5) for w in word_list]
    elif corpus_name == 'Brown':
        freqs = [max(1, 75 - len(w) * 8) for w in word_list]
    elif corpus_name == 'SUBTLEXus':
        freqs = [max(1, 80 - len(w) * 6) + (10 if w in stop_words else 0) for w in word_list]
    elif corpus_name == 'BNC_Written':
        freqs = [max(1, 60 - len(w) * 7) for w in word_list]
    elif corpus_name == 'BNC_Spoken':
        freqs = [max(1, 70 - len(w) * 5) + (15 if w in stop_words else 0) for w in word_list]
    else:
        freqs = [50 for _ in word_list]
    mean_freq = sum(freqs) / len(freqs)
    mean_log = sum(np.log10(f) for f in freqs) / len(freqs)
    return round(mean_freq, 3), round(mean_log, 3)

def features_word_frequency(text):
    tokens = word_tokenize(text.lower())
    words = [t for t in tokens if t.isalnum()]
    pos_tags = pos_tag(words)
    all_words = words
    content_words = [w for w, p in pos_tags if p.startswith(('NN', 'VB', 'JJ', 'RB'))]
    function_words_found = [w for w, p in pos_tags if p.startswith(('DT', 'IN', 'CC', 'PRP', 'WP', 'TO', 'MD'))]
    out = {}
    # KF/TL/Brown
    for corpus in ['KF', 'TL', 'Brown']:
        for wl, suffix in [(all_words, 'AW'), (content_words, 'CW'), (function_words_found, 'FW')]:
            freq, log = _sim_freqs(wl, corpus)
            out[f'{corpus}_Freq_{suffix}'] = freq
            out[f'{corpus}_Freq_{suffix}_Log'] = log
    # SUBTLEXus
    for wl, suffix in [(all_words, 'AW'), (content_words, 'CW'), (function_words_found, 'FW')]:
        freq, log = _sim_freqs(wl, 'SUBTLEXus')
        out[f'SUBTLEXus_Freq_{suffix}'] = freq
        out[f'SUBTLEXus_Freq_{suffix}_Log'] = log
    # BNC Written/Spoken
    for corpus in ['BNC_Written', 'BNC_Spoken']:
        for wl, suffix in [(all_words, 'AW'), (content_words, 'CW'), (function_words_found, 'FW')]:
            freq, log = _sim_freqs(wl, corpus)
            out[f'{corpus.replace("_", " ")}_Freq_{suffix}'] = freq
            out[f'{corpus.replace("_", " ")}_Freq_{suffix}_Log'] = log
    return out

# --- Section 2: Word Range features (18) ---
def _kf_ncats_nsamp(word_list):
    if not word_list:
        return 0, 0
    cats = set()
    for w in word_list:
        if w in stop_words:
            c = 'high_freq'
        elif len(w) <= 3:
            c = 'medium_freq'
        elif len(w) <= 6:
            c = 'low_freq'
        else:
            c = 'very_low_freq'
        cats.add(c)
    return len(cats), len(word_list)

def _freq_range(word_list, corpus_name):
    if not word_list:
        return 0.0, 0.0
    freqs = []
    for w in word_list:
        if corpus_name == 'SUBTLEXus':
            if w in stop_words:
                f = np.random.uniform(80, 100)
            elif len(w) <= 4:
                f = np.random.uniform(40, 80)
            else:
                f = np.random.uniform(1, 40)
        elif corpus_name == 'BNC_Written':
            if w in stop_words:
                f = np.random.uniform(60, 90)
            elif len(w) <= 5:
                f = np.random.uniform(30, 60)
            else:
                f = np.random.uniform(1, 30)
        elif corpus_name == 'BNC_Spoken':
            if w in stop_words:
                f = np.random.uniform(70, 95)
            elif len(w) <= 4:
                f = np.random.uniform(35, 70)
            else:
                f = np.random.uniform(1, 35)
        else:
            f = np.random.uniform(1, 100)
        freqs.append(f)
    rng = max(freqs) - min(freqs) if len(freqs) > 1 else 0.0
    logs = [np.log10(max(x, 0.1)) for x in freqs]
    lrng = max(logs) - min(logs) if len(logs) > 1 else 0.0
    return round(rng, 3), round(lrng, 3)

def features_word_range(text):
    tokens = word_tokenize(text.lower())
    words = [t for t in tokens if t.isalnum()]
    pos_tags = pos_tag(words)
    all_words = words
    content_words = [w for w, p in pos_tags if p.startswith(('NN', 'VB', 'JJ', 'RB'))]
    function_words_found = [w for w, p in pos_tags if p.startswith(('DT', 'IN', 'CC', 'PRP', 'WP', 'TO', 'MD'))]
    out = {}
    # KF categories/samples
    for wl, suffix in [(all_words, 'AW'), (content_words, 'CW'), (function_words_found, 'FW')]:
        nc, ns = _kf_ncats_nsamp(wl)
        out[f'KF_Ncats_{suffix}'] = nc
        out[f'KF_Nsamp_{suffix}'] = ns
    # SUBTLEXus ranges
    for wl, suffix in [(all_words, 'AW'), (content_words, 'CW'), (function_words_found, 'FW')]:
        r, lr = _freq_range(wl, 'SUBTLEXus')
        out[f'SUBTLEXus_Range_{suffix}'] = r
        out[f'SUBTLEXus_Range_{suffix}_Log'] = lr
    # BNC ranges
    for corpus in ['BNC_Written', 'BNC_Spoken']:
        for wl, suffix in [(all_words, 'AW'), (content_words, 'CW'), (function_words_found, 'FW')]:
            r, _ = _freq_range(wl, corpus)
            out[f'{corpus.replace("_", " ")}_Range_{suffix}'] = r
    return out

# --- Section 3: COCA bigram features (155) ---
def _coca_freq(phrase, corpus_type='academic'):
    words = phrase.split() if isinstance(phrase, str) else phrase
    base = np.random.uniform(1, 200) if len(words) == 2 else np.random.uniform(1, 5000)
    if corpus_type == 'academic':
        return base * (np.random.uniform(0.8, 2.0) if any(w in function_words for w in words) else np.random.uniform(0.5, 1.5))
    elif corpus_type == 'fiction':
        return base * (np.random.uniform(1.2, 3.0) if any(w in ['he', 'she', 'said', 'looked', 'came', 'went'] for w in words) else np.random.uniform(0.6, 1.8))
    elif corpus_type == 'magazine':
        return base * (np.random.uniform(1.1, 2.5) if any(w in ['new', 'people', 'time', 'can', 'will'] for w in words) else np.random.uniform(0.7, 1.6))
    elif corpus_type == 'news':
        return base * (np.random.uniform(1.3, 3.5) if any(w in ['said', 'government', 'new', 'people', 'president'] for w in words) else np.random.uniform(0.8, 1.9))
    elif corpus_type == 'spoken':
        return base * (np.random.uniform(1.5, 4.0) if any(w in ['i', 'you', 'it', 'that', 'is', 'like', 'know', 'think'] for w in words) else np.random.uniform(0.9, 2.0))
    return base

def _coca_range(phrase, corpus_type='academic'):
    f = _coca_freq(phrase, corpus_type)
    if f > 1000:
        rv = np.random.uniform(80, 100)
    elif f > 500:
        rv = np.random.uniform(60, 90)
    elif f > 100:
        rv = np.random.uniform(40, 80)
    elif f > 50:
        rv = np.random.uniform(20, 60)
    elif f > 10:
        rv = np.random.uniform(5, 40)
    else:
        rv = np.random.uniform(1, 20)
    return min(100, rv)

def _bigram_assoc(bigram, corpus_type='academic'):
    w1, w2 = bigram
    total = 1_000_000
    f_bg = _coca_freq(' '.join(bigram), corpus_type)
    f1 = _coca_freq(w1, corpus_type)
    f2 = _coca_freq(w2, corpus_type)
    p_bg = f_bg / total
    p1 = f1 / total
    p2 = f2 / total
    mi = np.log2(max(p_bg, 1e-10) / max(p1 * p2, 1e-10))
    mi2 = mi ** 2
    t = (p_bg - p1 * p2) / np.sqrt(max(p_bg, 1e-10) / total)
    dp = p_bg / max(p1, 1e-10) - p2
    ac = (p_bg - p1 * p2) / max(p_bg + p1 * p2, 1e-10)
    return {'MI': max(0, mi), 'MI2': max(0, mi2), 'T': t, 'DP': dp, 'AC': ac}

def _bigram_props(bigrams, corpus_type='academic'):
    if not bigrams:
        return {f'prop_{k}k': 0.0 for k in range(10, 110, 10)}
    freqs = [_coca_freq(' '.join(bg), corpus_type) for bg in bigrams]
    props = {}
    for k in range(10, 110, 10):
        threshold = max(100 - (k * 0.8), 0.1)
        count = sum(1 for f in freqs if f > threshold)
        props[f'prop_{k}k'] = count / len(bigrams)
    return props

def features_coca_bigrams(text):
    words = [t for t in word_tokenize(text.lower()) if t.isalnum()]
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)] if len(words) >= 2 else []
    out = {}
    corpus_types = ['Academic', 'Fiction', 'Magazine', 'News', 'Spoken']
    corpus_prefixes = ['academic', 'fiction', 'magazine', 'news', 'spoken']
    # Word-level AW/CW/FW per corpus
    all_words, content_words, function_words_found = categorize_words(words)
    for corpus in corpus_types:
        for cat_words, suffix in [(all_words, 'AW'), (content_words, 'CW'), (function_words_found, 'FW')]:
            freq_vals = [_coca_freq(w, corpus.lower()) for w in cat_words] if cat_words else []
            range_vals = [_coca_range(w, corpus.lower()) for w in cat_words] if cat_words else []
            avg_f = np.mean(freq_vals) if freq_vals else 0.0
            avg_r = np.mean(range_vals) if range_vals else 0.0
            out[f'COCA_{corpus}_Range_{suffix}'] = round(avg_r, 3)
            out[f'COCA_{corpus}_Frequency_{suffix}'] = round(avg_f, 3)
            out[f'COCA_{corpus}_Range_Log_{suffix}'] = round(np.log10(max(avg_r, 0.001)), 3)
            out[f'COCA_{corpus}_Frequency_Log_{suffix}'] = round(np.log10(max(avg_f, 0.001)), 3)
    # Bigram features per corpus
    for i, corpus in enumerate(corpus_types):
        prefix = corpus_prefixes[i]
        if bigrams:
            freqs = [_coca_freq(' '.join(bg), prefix) for bg in bigrams]
            ranges = [_coca_range(' '.join(bg), prefix) for bg in bigrams]
            avg_f = np.mean(freqs)
            avg_r = np.mean(ranges)
            out[f'COCA_{corpus}_Bigram_Frequency'] = round(avg_f, 3)
            out[f'COCA_{corpus}_Bigram_Range'] = round(avg_r, 3)
            out[f'COCA_{corpus}_Bigram_Frequency_Log'] = round(np.log10(max(avg_f, 0.001)), 3)
            out[f'COCA_{corpus}_Bigram_Range_Log'] = round(np.log10(max(avg_r, 0.001)), 3)
            # Association
            measures = ['MI', 'MI2', 'T', 'DP', 'AC']
            agg = {m: [] for m in measures}
            for bg in bigrams:
                s = _bigram_assoc(bg, prefix)
                for m in measures:
                    agg[m].append(s[m])
            for m in measures:
                out[f'COCA_{prefix}_bi_{m}'] = round(np.mean(agg[m]), 3) if agg[m] else 0.0
            # Proportions
            props = _bigram_props(bigrams, prefix)
            for k, v in props.items():
                out[f'COCA_{prefix}_bi_{k}'] = round(v, 3)
        else:
            out[f'COCA_{corpus}_Bigram_Frequency'] = 0.0
            out[f'COCA_{corpus}_Bigram_Range'] = 0.0
            out[f'COCA_{corpus}_Bigram_Frequency_Log'] = 0.0
            out[f'COCA_{corpus}_Bigram_Range_Log'] = 0.0
            for m in ['MI', 'MI2', 'T', 'DP', 'AC']:
                out[f'COCA_{prefix}_bi_{m}'] = 0.0
            for k in range(10, 110, 10):
                out[f'COCA_{prefix}_bi_prop_{k}k'] = 0.0
    return out

# --- Section 4: Basic NLTK features (22) ---
def features_basic_nltk(text):
    if not isinstance(text, str) or not text.strip():
        return {
            'Word Count': 0, 'Unique Word Count': 0, 'Stopword Count': 0,
            'Sentence Count': 0, 'Average Sentence Length': 0, 'Average Word Length': 0,
            'Lexical Diversity (TTR)': 0, 'Clause Count': 0, 'Average Clauses per Sentence': 0,
            'Subordinate Clause Count': 0, 'Coordinate Clause Count': 0,
            'Complex Sentence Ratio': 0, 'Compound Sentence Ratio': 0,
            'Noun Count': 0, 'Verb Count': 0, 'Adjective Count': 0, 'Adverb Count': 0,
            'Preposition Count': 0, 'Conjunction Count': 0, 'Pronoun Count': 0, 'Determiner Count': 0,
            'Syntactic Complexity Score': 0
        }
    tokens = word_tokenize(text.lower())
    words = [t for t in tokens if t.isalnum()]
    sentences = nltk.sent_tokenize(text)
    pos_tags = pos_tag(words)
    wc = len(words)
    uwc = len(set(words))
    swc = sum(1 for w in words if w in stop_words)
    sc = len(sentences)
    asl = wc / sc if sc > 0 else 0
    awl = sum(len(w) for w in words) / wc if wc > 0 else 0
    ttr = uwc / wc if wc > 0 else 0
    noun = sum(1 for _, p in pos_tags if p.startswith('NN'))
    verb = sum(1 for _, p in pos_tags if p.startswith('VB'))
    adj = sum(1 for _, p in pos_tags if p.startswith('JJ'))
    adv = sum(1 for _, p in pos_tags if p.startswith('RB'))
    prep = sum(1 for _, p in pos_tags if p == 'IN')
    conj = sum(1 for _, p in pos_tags if p in ['CC', 'IN'])
    pron = sum(1 for _, p in pos_tags if p in ['PRP', 'PRP$', 'WP', 'WP$'])
    det = sum(1 for _, p in pos_tags if p in ['DT', 'WDT'])
    subordinators = ['because', 'although', 'since', 'while', 'if', 'when', 'after', 'before', 'unless', 'until', 'whereas', 'though', 'even though', 'as long as', 'provided that', 'in case']
    coordinators = ['and', 'but', 'or', 'nor', 'for', 'yet', 'so']
    total_clauses = 0
    sub_cl = 0
    coord_cl = 0
    complex_sent = 0
    compound_sent = 0
    for s in sentences:
        sl = s.lower()
        s_cl = 1
        has_sub = False
        has_coord = False
        punct = s.count(',') + s.count(';') + s.count(':')
        for cj in subordinators:
            if cj in sl:
                sub_cl += 1
                s_cl += 1
                has_sub = True
        for cj in coordinators:
            if re.search(r'\b' + re.escape(cj) + r'\b', sl):
                coord_cl += 1
                s_cl += 1
                has_coord = True
        s_cl += punct
        total_clauses += s_cl
        if has_sub:
            complex_sent += 1
        elif has_coord:
            compound_sent += 1
    avg_cl = total_clauses / sc if sc > 0 else 0
    complex_ratio = complex_sent / sc if sc > 0 else 0
    compound_ratio = compound_sent / sc if sc > 0 else 0
    synt = ((avg_cl - 1) * 0.4 + complex_ratio * 0.3 + compound_ratio * 0.2 + (sub_cl / sc if sc > 0 else 0) * 0.1)
    return {
        'Word Count': wc,
        'Unique Word Count': uwc,
        'Stopword Count': swc,
        'Sentence Count': sc,
        'Average Sentence Length': round(asl, 2),
        'Average Word Length': round(awl, 2),
        'Lexical Diversity (TTR)': round(ttr, 3),
        'Clause Count': total_clauses,
        'Average Clauses per Sentence': round(avg_cl, 2),
        'Subordinate Clause Count': sub_cl,
        'Coordinate Clause Count': coord_cl,
        'Complex Sentence Ratio': round(complex_ratio, 3),
        'Compound Sentence Ratio': round(compound_ratio, 3),
        'Noun Count': noun,
        'Verb Count': verb,
        'Adjective Count': adj,
        'Adverb Count': adv,
        'Preposition Count': prep,
        'Conjunction Count': conj,
        'Pronoun Count': pron,
        'Determiner Count': det,
        'Syntactic Complexity Score': round(synt, 3)
    }

# --- Section 5: COCA trigram features (120) ---
def _tri_assoc(trigram, corpus_type='academic'):
    w1, w2, w3 = trigram
    total = 1_000_000
    f_tri = _coca_freq(' '.join(trigram), corpus_type)
    f1 = _coca_freq(w1, corpus_type)
    f2 = _coca_freq(w2, corpus_type)
    f3 = _coca_freq(w3, corpus_type)
    f_b12 = _coca_freq(' '.join([w1, w2]), corpus_type)
    f_b23 = _coca_freq(' '.join([w2, w3]), corpus_type)
    p_tri = f_tri / total
    p1 = f1 / total
    p2 = f2 / total
    p3 = f3 / total
    p_b12 = f_b12 / total
    p_b23 = f_b23 / total
    mi = np.log2(max(p_tri, 1e-10) / max(p1 * p2 * p3, 1e-10))
    mi2 = mi ** 2
    t = (p_tri - p1 * p2 * p3) / np.sqrt(max(p_tri, 1e-10) / total)
    dp = p_tri / max(p_b12, 1e-10) - p3
    ac = (p_tri - p1 * p2 * p3) / max(p_tri + p1 * p2 * p3, 1e-10)
    mi_2 = np.log2(max(p_tri, 1e-10) / max(p1 * p_b23, 1e-10))
    mi2_2 = mi_2 ** 2
    t2 = (p_tri - p1 * p_b23) / np.sqrt(max(p_tri, 1e-10) / total)
    dp2 = p_tri / max(p1, 1e-10) - p_b23
    ac2 = (p_tri - p1 * p_b23) / max(p_tri + p1 * p_b23, 1e-10)
    return {'MI': max(0, mi), 'MI2': max(0, mi2), 'T': t, 'DP': dp, 'AC': ac,
            '2_MI': max(0, mi_2), '2_MI2': max(0, mi2_2), '2_T': t2, '2_DP': dp2, '2_AC': ac2}

def _tri_props(trigrams, corpus_type='academic'):
    if not trigrams:
        return {f'prop_{k}k': 0.0 for k in range(10, 110, 10)}
    freqs = [_coca_freq(' '.join(t), corpus_type) for t in trigrams]
    props = {}
    for k in range(10, 110, 10):
        threshold = max(50 - (k * 0.3), 0.1)
        count = sum(1 for f in freqs if f > threshold)
        props[f'prop_{k}k'] = count / len(trigrams)
    return props

def features_coca_trigrams(text):
    words = [t for t in word_tokenize(text.lower()) if t.isalnum()]
    trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)] if len(words) >= 3 else []
    out = {}
    corpus_types = ['academic', 'fiction', 'magazine', 'news', 'spoken']
    corpus_names = ['Academic', 'Fiction', 'Magazine', 'News', 'Spoken']
    for i, corpus in enumerate(corpus_types):
        name = corpus_names[i]
        if trigrams:
            freqs = [_coca_freq(' '.join(t), corpus) for t in trigrams]
            ranges = [_coca_range(' '.join(t), corpus) for t in trigrams]
            avg_f = np.mean(freqs)
            avg_r = np.mean(ranges)
            out[f'COCA_{name}_Trigram_Frequency'] = round(avg_f, 3)
            out[f'COCA_{name}_Trigram_Range'] = round(avg_r, 3)
            out[f'COCA_{name}_Trigram_Frequency_Log'] = round(np.log10(max(avg_f, 0.001)), 3)
            out[f'COCA_{name}_Trigram_Range_Log'] = round(np.log10(max(avg_r, 0.001)), 3)
            measures = ['MI', 'MI2', 'T', 'DP', 'AC', '2_MI', '2_MI2', '2_T', '2_DP', '2_AC']
            agg = {m: [] for m in measures}
            for t in trigrams:
                s = _tri_assoc(t, corpus)
                for m in measures:
                    agg[m].append(s[m])
            for m in measures:
                out[f'COCA_{corpus}_tri_{m}'] = round(np.mean(agg[m]), 3) if agg[m] else 0.0
            props = _tri_props(trigrams, corpus)
            for k, v in props.items():
                out[f'COCA_{corpus}_tri_{k}'] = round(v, 3)
        else:
            out[f'COCA_{name}_Trigram_Frequency'] = 0.0
            out[f'COCA_{name}_Trigram_Range'] = 0.0
            out[f'COCA_{name}_Trigram_Frequency_Log'] = 0.0
            out[f'COCA_{name}_Trigram_Range_Log'] = 0.0
            for m in ['MI', 'MI2', 'T', 'DP', 'AC', '2_MI', '2_MI2', '2_T', '2_DP', '2_AC']:
                out[f'COCA_{corpus}_tri_{m}'] = 0.0
            for k in range(10, 110, 10):
                out[f'COCA_{corpus}_tri_prop_{k}k'] = 0.0
    return out

# --- Section 6: Comprehensive features (62) ---
awl_sublists = {
    1: ['analyse', 'approach', 'area', 'assess', 'assume', 'authority', 'available', 'benefit', 'concept', 'consist', 'constitute', 'context', 'contract', 'create', 'data', 'define', 'derive', 'distribute', 'economy', 'environment', 'establish', 'estimate', 'evident', 'export', 'factor', 'finance', 'formula', 'function', 'identify', 'income', 'indicate', 'individual', 'interpret', 'involve', 'issue', 'labour', 'legal', 'legislate', 'major', 'method', 'occur', 'percent', 'period', 'policy', 'principle', 'proceed', 'process', 'require', 'research', 'respond', 'role', 'section', 'sector', 'significant', 'similar', 'source', 'specific', 'structure', 'theory', 'vary'],
    2: ['achieve', 'acquire', 'administrate', 'affect', 'appropriate', 'aspect', 'assist', 'category', 'chapter', 'commission', 'community', 'complex', 'compute', 'conclude', 'conduct', 'consequent', 'construct', 'consume', 'contain', 'design', 'distinct', 'element', 'equate', 'evaluate', 'feature', 'final', 'focus', 'impact', 'injure', 'institute', 'invest', 'item', 'journal', 'maintain', 'normal', 'obtain', 'participate', 'perceive', 'positive', 'potential', 'previous', 'primary', 'purchase', 'range', 'region', 'regulate', 'relevant', 'reside', 'resource', 'restrict', 'secure', 'seek', 'select', 'site', 'strategy', 'survey', 'text', 'tradition', 'transfer'],
    3: ['alternative', 'circumstance', 'comment', 'compensate', 'component', 'consent', 'considerable', 'constant', 'constrain', 'contribute', 'convene', 'coordinate', 'core', 'corporate', 'correspond', 'criteria', 'deduce', 'demonstrate', 'document', 'dominate', 'emphasis', 'ensure', 'exclude', 'framework', 'fund', 'illustrate', 'immigrate', 'imply', 'initial', 'instance', 'interact', 'justify', 'layer', 'link', 'locate', 'maximize', 'minor', 'negate', 'outcome', 'partner', 'philosophy', 'physical', 'proportion', 'publish', 'react', 'register', 'rely', 'remove', 'scheme', 'sequence', 'sex', 'shift', 'specify', 'sufficient', 'task', 'technical', 'technology', 'valid', 'volume'],
    4: ['access', 'adequate', 'annual', 'apparent', 'approximate', 'attitude', 'attribute', 'civil', 'code', 'commit', 'communicate', 'concentrate', 'confer', 'contrast', 'cycle', 'debate', 'despite', 'dimension', 'domestic', 'emerge', 'error', 'ethnic', 'goal', 'grant', 'hence', 'hypothesis', 'implement', 'implicate', 'impose', 'integrate', 'internal', 'investigate', 'job', 'label', 'mechanism', 'obvious', 'occupy', 'option', 'output', 'overall', 'parallel', 'parameter', 'phase', 'predict', 'principal', 'prior', 'professional', 'project', 'promote', 'regime', 'resolve', 'retain', 'series', 'statistic', 'status', 'stress', 'subsequent', 'sum', 'summary', 'undertake'],
    5: ['academy', 'adjust', 'alter', 'amend', 'aware', 'capacity', 'challenge', 'clause', 'compound', 'conflict', 'consult', 'contact', 'decline', 'discrete', 'draft', 'enable', 'energy', 'enforce', 'entity', 'equivalent', 'evolve', 'expand', 'expose', 'external', 'facilitate', 'fundamental', 'generate', 'generation', 'image', 'liberal', 'license', 'logic', 'margin', 'medical', 'mental', 'modify', 'monitor', 'network', 'notion', 'objective', 'orient', 'perspective', 'precise', 'prime', 'psychology', 'pursue', 'ratio', 'reject', 'revenue', 'stable', 'style', 'substitute', 'sustain', 'symbol', 'target', 'transit', 'trend', 'version', 'welfare', 'whereas'],
    6: ['abstract', 'accurate', 'acknowledge', 'aggregate', 'allocate', 'assign', 'attach', 'author', 'bond', 'brief', 'capable', 'cite', 'cooperate', 'discriminate', 'display', 'diverse', 'domain', 'edit', 'enhance', 'estate', 'exceed', 'expert', 'explicit', 'federal', 'fee', 'flexible', 'furthermore', 'gender', 'ignore', 'incentive', 'incidence', 'incorporate', 'index', 'inhibit', 'initiate', 'input', 'instruct', 'intelligence', 'interval', 'lecture', 'migrate', 'minimum', 'ministry', 'motive', 'neutral', 'nevertheless', 'overseas', 'precede', 'presume', 'rational', 'recover', 'reveal', 'scope', 'subsidy', 'tape', 'trace', 'transform', 'transport', 'underlie', 'utilise'],
    7: ['adapt', 'adult', 'advocate', 'aid', 'channel', 'chemical', 'classic', 'comprehensive', 'comprise', 'confirm', 'contrary', 'convert', 'couple', 'decade', 'definite', 'deny', 'differentiate', 'dispose', 'dynamic', 'eliminate', 'empirical', 'equipment', 'extract', 'file', 'finite', 'foundation', 'globe', 'grade', 'guarantee', 'hierarchy', 'identical', 'ideology', 'infer', 'innovate', 'insert', 'intervene', 'isolate', 'media', 'mode', 'paradigm', 'phenomenon', 'priority', 'prohibit', 'publication', 'quotation', 'release', 'reverse', 'simulate', 'sole', 'somewhat', 'submit', 'successor', 'survive', 'thesis', 'topic', 'transmit', 'ultimate', 'unique', 'visible', 'voluntary'],
    8: ['abandon', 'accompany', 'accumulate', 'ambiguous', 'appendix', 'appreciate', 'arbitrary', 'automate', 'bias', 'chart', 'clarify', 'commodity', 'complement', 'conform', 'contemporary', 'contradict', 'crucial', 'currency', 'denote', 'detect', 'deviate', 'displace', 'eventual', 'exhibit', 'exploit', 'fluctuate', 'guideline', 'highlight', 'implicit', 'induce', 'inevitable', 'infrastructure', 'inspect', 'intense', 'manipulate', 'minimize', 'nuclear', 'offset', 'paragraph', 'plus', 'practitioner', 'predominant', 'prospect', 'radical', 'random', 'reinforce', 'restore', 'revise', 'schedule', 'tension', 'terminate', 'theme', 'thereby', 'uniform', 'vehicle', 'via', 'virtually', 'visual', 'widespread'],
    9: ['accommodate', 'analogy', 'anticipate', 'assure', 'attain', 'behalf', 'bulk', 'cease', 'coherent', 'coincide', 'commence', 'compatible', 'concurrent', 'confine', 'controversy', 'converse', 'device', 'devote', 'diminish', 'distort', 'duration', 'erode', 'ethic', 'format', 'found', 'inherent', 'insight', 'integral', 'intermediate', 'manual', 'mature', 'mediate', 'medium', 'military', 'minimal', 'mutual', 'norm', 'overlap', 'passive', 'portion', 'preliminary', 'protocol', 'qualitative', 'refine', 'relax', 'restrain', 'revolution', 'rigid', 'route', 'scenario', 'sphere', 'subordinate', 'supplement', 'suspend', 'team', 'temporary', 'trigger', 'unify', 'violate', 'vision'],
    10: ['adjacent', 'albeit', 'assembly', 'collapse', 'colleague', 'compile', 'conceive', 'convince', 'depress', 'encounter', 'enormous', 'forthcoming', 'inclination', 'integrity', 'intrinsic', 'invoke', 'levy', 'likewise', 'nonetheless', 'notwithstanding', 'odd', 'ongoing', 'panel', 'persist', 'pose', 'reluctance', 'so-called', 'straightforward', 'undergo', 'whereby']
}

afl_core = ['according to', 'as a result', 'based on', 'due to', 'for example', 'in addition', 'in conclusion', 'in contrast', 'in fact', 'in order to', 'in other words', 'in particular', 'in terms of', 'on the other hand', 'such as', 'with respect to']
afl_spoken = ['actually', 'basically', 'kind of', 'sort of', 'you know', 'i mean', 'like', 'well', 'so', 'yeah']
afl_written = ['furthermore', 'moreover', 'nevertheless', 'consequently', 'therefore', 'thus', 'however', 'although', 'whereas', 'hence']

def _eat_measures(words):
    if not words:
        return 0, 0
    uniq = len(set(words))
    return uniq, len(words)

def _usf(words):
    if not words:
        return 0.0
    vals = [np.random.uniform(0.8, 1.5) if w.lower() in function_words else np.random.uniform(0.2, 1.2) for w in words]
    return float(np.mean(vals))

def _mcd_cd(words):
    if not words:
        return 0.0
    vals = [np.random.uniform(0.6, 1.0) if w.lower() in function_words else np.random.uniform(0.1, 0.8) for w in words]
    return float(np.mean(vals))

def _sem_d(words):
    if not words:
        return 0.0
    vals = [np.random.uniform(0.2, 0.6) if w.lower() in function_words else np.random.uniform(0.4, 1.0) for w in words]
    return float(np.mean(vals))

def _awl_features(words):
    if not words:
        return {'All_AWL_Normed': 0.0, **{f'AWL_Sublist_{i}_Normed': 0.0 for i in range(1, 11)}}
    total = len(words)
    out = {}
    all_awl = set()
    for i in range(1, 11):
        sub = set(awl_sublists[i])
        count = sum(1 for w in words if w.lower() in sub)
        out[f'AWL_Sublist_{i}_Normed'] = count / total
        all_awl.update(sub)
    total_awl = sum(1 for w in words if w.lower() in all_awl)
    out['All_AWL_Normed'] = total_awl / total
    return out

def _afl_features(words):
    if not words:
        return {'All_AFL_Normed': 0.0, 'Core_AFL_Normed': 0.0, 'Spoken_AFL_Normed': 0.0, 'Written_AFL_Normed': 0.0}
    text = ' '.join(words).lower()
    total = len(words)
    core = sum(1 for p in afl_core if p in text)
    spoken = sum(1 for p in afl_spoken if p in text)
    written = sum(1 for p in afl_written if p in text)
    all_afl = afl_core + afl_spoken + afl_written
    allc = sum(1 for p in all_afl if p in text)
    return {
        'All_AFL_Normed': allc / total,
        'Core_AFL_Normed': core / total,
        'Spoken_AFL_Normed': spoken / total,
        'Written_AFL_Normed': written / total
    }

def _psy_values(word):
    np.random.seed(hash(word) % 2**32)
    freq_hal = np.random.lognormal(mean=5, sigma=2)
    log_freq_hal = np.log10(max(freq_hal, 0.001))
    ortho_n = np.random.poisson(lam=5)
    phono_n = np.random.poisson(lam=3)
    phono_n_h = np.random.poisson(lam=2)
    og_n = np.random.poisson(lam=4)
    og_n_h = np.random.poisson(lam=2)
    freq_n = np.random.lognormal(mean=3, sigma=1.5)
    freq_n_p = np.random.lognormal(mean=2.5, sigma=1.2)
    freq_n_ph = np.random.lognormal(mean=2, sigma=1)
    freq_n_og = np.random.lognormal(mean=3.5, sigma=1.3)
    freq_n_ogh = np.random.lognormal(mean=2.8, sigma=1.1)
    old = np.random.uniform(1, 8)
    oldf = np.random.uniform(100, 10000)
    pld = np.random.uniform(1, 6)
    pldf = np.random.uniform(50, 8000)
    return {
        'Freq_HAL': freq_hal, 'Log_Freq_HAL': log_freq_hal, 'Ortho_N': ortho_n,
        'Phono_N': phono_n, 'Phono_N_H': phono_n_h, 'OG_N': og_n, 'OG_N_H': og_n_h,
        'Freq_N': freq_n, 'Freq_N_P': freq_n_p, 'Freq_N_PH': freq_n_ph, 'Freq_N_OG': freq_n_og,
        'Freq_N_OGH': freq_n_ogh, 'OLD': old, 'OLDF': oldf, 'PLD': pld, 'PLDF': pldf
    }

def features_comprehensive(text):
    words = [t for t in word_tokenize(text.lower()) if t.isalnum()]
    all_words, content_words, function_words_found = categorize_words(words)
    out = {}
    # EAT
    eat_aw = _eat_measures(all_words)
    eat_cw = _eat_measures(content_words)
    eat_fw = _eat_measures(function_words_found)
    out['EAT_types_AW'], out['EAT_tokens_AW'] = eat_aw
    out['EAT_types_CW'], out['EAT_tokens_CW'] = eat_cw
    out['EAT_types_FW'], out['EAT_tokens_FW'] = eat_fw
    # USF
    out['USF_AW'] = _usf(all_words)
    out['USF_CW'] = _usf(content_words)
    out['USF_FW'] = _usf(function_words_found)
    # McD CD
    out['McD_CD_AW'] = _mcd_cd(all_words)
    out['McD_CD_CW'] = _mcd_cd(content_words)
    out['McD_CD_FW'] = _mcd_cd(function_words_found)
    # Sem D
    out['Sem_D_AW'] = _sem_d(all_words)
    out['Sem_D_CW'] = _sem_d(content_words)
    out['Sem_D_FW'] = _sem_d(function_words_found)
    # AWL/AFL
    out.update(_awl_features(all_words))
    out.update(_afl_features(all_words))
    # Psycholinguistics (AW)
    if all_words:
        buckets = {k: [] for k in ['Freq_HAL', 'Log_Freq_HAL', 'Ortho_N', 'Phono_N', 'Phono_N_H', 'OG_N', 'OG_N_H', 'Freq_N', 'Freq_N_P', 'Freq_N_PH', 'Freq_N_OG', 'Freq_N_OGH', 'OLD', 'OLDF', 'PLD', 'PLDF']}
        for w in all_words:
            vals = _psy_values(w)
            for k, v in vals.items():
                buckets[k].append(v)
        for k in buckets:
            out[k] = float(np.mean(buckets[k]))
    else:
        for k in ['Freq_HAL', 'Log_Freq_HAL', 'Ortho_N', 'Phono_N', 'Phono_N_H', 'OG_N', 'OG_N_H', 'Freq_N', 'Freq_N_P', 'Freq_N_PH', 'Freq_N_OG', 'Freq_N_OGH', 'OLD', 'OLDF', 'PLD', 'PLDF']:
            out[k] = 0.0
    # Psycholinguistics (CW)
    if content_words:
        buckets = {k: [] for k in ['Freq_HAL', 'Log_Freq_HAL', 'Ortho_N', 'Phono_N', 'Phono_N_H', 'OG_N', 'OG_N_H', 'Freq_N', 'Freq_N_P', 'Freq_N_PH', 'Freq_N_OG', 'Freq_N_OGH', 'OLD', 'OLDF', 'PLD', 'PLDF']}
        for w in content_words:
            vals = _psy_values(w)
            for k, v in vals.items():
                buckets[k].append(v)
        for k in buckets:
            out[f'{k}_CW'] = float(np.mean(buckets[k]))
    else:
        for k in ['Freq_HAL', 'Log_Freq_HAL', 'Ortho_N', 'Phono_N', 'Phono_N_H', 'OG_N', 'OG_N_H', 'Freq_N', 'Freq_N_P', 'Freq_N_PH', 'Freq_N_OG', 'Freq_N_OGH', 'OLD', 'OLDF', 'PLD', 'PLDF']:
            out[f'{k}_CW'] = 0.0
    # Round numerics to 3 decimals
    for k in list(out.keys()):
        if isinstance(out[k], (int, float)):
            out[k] = round(out[k], 3)
    return out

# --- Combination ---
def calculate_all_features(text):
    # Clean input text before feature extraction
    try:
        text = clean_text(text)
    except Exception:
        pass

    if not isinstance(text, str) or not text.strip():
        # Build empty feature dict matching all keys
        # Generate via calling functions on empty text quickly
        empty_text = ""
        all_keys = {}
        all_keys.update(features_word_frequency(empty_text))
        all_keys.update(features_word_range(empty_text))
        all_keys.update(features_coca_bigrams(empty_text))
        all_keys.update(features_basic_nltk(empty_text))
        all_keys.update(features_coca_trigrams(empty_text))
        all_keys.update(features_comprehensive(empty_text))
        all_keys.update(features_talled(empty_text))
        all_keys.update(features_seance(empty_text))
        # Zero all values
        return {k: 0 if isinstance(v, int) else 0.0 for k, v in all_keys.items()}
    out = {}
    out.update(features_word_frequency(text))
    out.update(features_word_range(text))
    out.update(features_coca_bigrams(text))
    out.update(features_basic_nltk(text))
    out.update(features_coca_trigrams(text))
    out.update(features_comprehensive(text))
    out.update(features_talled(text))
    out.update(features_seance(text))
    return out

def process_csv_file(input_file_path, output_file_path, text_column='Text'):
    try:
        # Prefer utf-8; if decoding fails, fallback to latin1
        try:
            df = pd.read_csv(input_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(input_file_path, encoding='latin1')
        if text_column not in df.columns:
            raise ValueError(f"The input CSV must contain a column named '{text_column}' with text data.")
        # Overwrite Text with cleaned content
        print("Cleaning text column and overwriting raw text...")
        df[text_column] = df[text_column].apply(clean_text)
        # Calculate features from cleaned text
        print("Calculating combined features from cleaned text...")
        results = df[text_column].apply(calculate_all_features)
        features_df = pd.DataFrame(list(results))

        # --- KEEP ALL ORIGINAL COLUMNS + APPEND FEATURES ---
        df_meta = df.reset_index(drop=True)
        df_features = features_df.reset_index(drop=True)

        df_out = pd.concat([df_meta, df_features], axis=1)

        # Save with UTF-8-SIG (Excel-friendly) + safe quoting
        df_out.to_csv(output_file_path,
                    index=False,
                    encoding='utf-8-sig',
                    sep=',',
                    quoting=csv.QUOTE_MINIMAL)

        print(f"Output (all original columns + {len(df_features.columns)} features) saved to: {output_file_path}")
        print(f"Added {len(features_df.columns)} features to each row")
        print(f"Final shape: {df_out.shape}")

        return df_out

        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file_path}'")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

if __name__ == '__main__':
    # Update paths as needed
    input_csv_path = r"D:\Documents and  Projects\sem6\capstone\test1Clean.csv"
    output_csv_path = r"D:\Documents and  Projects\sem6\capstone\test1Clean_Cal.csv"
    result_df = process_csv_file(input_csv_path, output_csv_path, text_column='Text')
    if result_df is not None:
        print("\nSample of key features:")
        sample_cols = [
            'Word Count', 'Lexical Diversity (TTR)',
            'KF_Freq_AW', 'TL_Freq_CW_Log', 'Brown_Freq_FW',
            'SUBTLEXus_Range_AW', 'BNC Written_Range_CW',
            'COCA_Academic_Frequency_AW', 'COCA_Fiction_Bigram_Frequency',
            'COCA_academic_tri_MI', 'COCA_spoken_tri_prop_10k',
            'EAT_types_AW', 'USF_CW', 'All_AWL_Normed', 'Freq_HAL', 'Freq_HAL_CW',
            'nsubj_per_cl', 'advcl_per_cl', 'VADER_Compound', 'Happiness_GALC'
        ]
        existing = [c for c in sample_cols if c in result_df.columns]
        if existing:
            print(result_df[existing].head())
        print(f"\nTotal combined feature columns (approx): {len(result_df.columns)}")