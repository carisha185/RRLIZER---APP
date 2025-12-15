from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import pandas as pd
import numpy as np
import re
import math
import io
import base64
import os
import time
from io import BytesIO
from contextlib import redirect_stdout
from django.conf import settings
from multiprocessing import Pool, cpu_count
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import joblib
import json
import pickle


try:
    from .slang_dictionary import KAMUS_SLANG
except ImportError:
    KAMUS_SLANG = {}

#download resourcesnya NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

#cache instance untuk proses lebih cepat
_stemmer = None
_stopwords_id = None
_kamus_slang = None

# ===== PRE-COMPILED REGEX PATTERNS (GLOBAL CACHE) =====
_REGEX_HASHTAG = re.compile(r'#[A-Za-z0-9]+')
_REGEX_URL = re.compile(r"http\S+")
_REGEX_NUMBERS = re.compile(r'[0-9]+')
_REGEX_SPECIAL = re.compile(r"[-()\"#/@;:<>{}'+=~|.!?,_]")
_REGEX_SINGLE = re.compile(r"\b[a-zA-Z]\b")
_REGEX_REPEATED = re.compile(r'(.)\1{2,}')
_REGEX_SPLIT_SENTENCE = re.compile(r'[.,!?]+')
_REGEX_MULTI_SPACE = re.compile(r'\s+')

#kalimat negasi yg tidak dihapus dari daftar stopwords
SENTIMENT_BEARING_WORDS = {
    'gak', 'ga', 'tidak', 'nggak', 'ngga', 'enggak',
    'jelek', 'buruk', 'payah', 'kecewa', 'mengecewakan',
    'hancur', 'rusak', 'cacat', 'bau','lama', 'kurang'
}

def get_stemmer():
    global _stemmer
    if _stemmer is None:
        factory = StemmerFactory()
        _stemmer = factory.create_stemmer()
    return _stemmer

def get_stopwords():
    """
    Custom stopwords yang exclude sentiment bearing words.
    Ini penting untuk sentiment analysis!
    """
    global _stopwords_id
    if _stopwords_id is None:
        nltk_stopwords = set(stopwords.words('indonesian'))
        _stopwords_id = nltk_stopwords - SENTIMENT_BEARING_WORDS
    return _stopwords_id

def get_kamus_slang():
    global _kamus_slang
    if _kamus_slang is None:
        try:
            kamus_slang_df = pd.read_csv('full.csv')
            _kamus_slang = dict(zip(kamus_slang_df['transformed'], kamus_slang_df['original-for']))
        except FileNotFoundError:
            _kamus_slang = KAMUS_SLANG if KAMUS_SLANG else {}
    return _kamus_slang

def _read_any(fileobj):
    """Baca CSV/XLSX dari request.FILES."""
    name = fileobj.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(fileobj, encoding='utf-8')
    if name.endswith('.xlsx'):
        return pd.read_excel(fileobj)
    raise ValueError('Unsupported file (use .csv or .xlsx)')

def _fig_to_b64img():
    """Simpan figure aktif ke PNG base64."""
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# pembangunan vektor fitur
class FeatureVectorBuilder:
    """Builder untuk mengubah sentences jadi feature vectors."""
    
    def __init__(self, positive_keywords, negative_keywords):
        """
        positive_keywords: list/set of positive keywords
        negative_keywords: list/set of negative keywords
        """
        self.all_keywords = list(set(positive_keywords) | set(negative_keywords))
        self.keyword_to_idx = {kw: i for i, kw in enumerate(self.all_keywords)}

    def build_vector(self, sentence):
        """Convert sentence ke feature vector (1D numpy array)."""
        vec = np.zeros(len(self.all_keywords), dtype=float)
        for w in str(sentence).lower().split():
            idx = self.keyword_to_idx.get(w)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    def build_vectors(self, X):
        """Convert list of sentences ke feature vectors (2D numpy array)."""
        if len(X) == 0:
            return np.empty((0, len(self.all_keywords)))
        return np.vstack([self.build_vector(s) for s in X])


#load model SVM
SVMODEL_PATH = os.path.join(settings.BASE_DIR, 'sentiment_app', 'skmodel.pkl')

MODEL = None
BUILDER = None
MODEL_META = {}

def _reconstruct_builder_from_dict(builder_dict):
    """
    Buat FeatureVectorBuilder object dari dict (untuk sklearn format).
    Dict harus punya keys: 'keywords' dan 'keyword_to_idx'
    """
    try:
        keywords = builder_dict.get('keywords', [])
        keyword_to_idx = builder_dict.get('keyword_to_idx', {})
        
        #object dummy
        builder = FeatureVectorBuilder([], [])
        builder.all_keywords = list(keywords)
        builder.keyword_to_idx = keyword_to_idx
        
        return builder
    except Exception as e:
        print(f"Error rekonstruksi builder: {e}")
        return None


def _load_sklearn_svm_model():
    """
    Load sklearn SVM model dari file pickle.
    {
        'model': <sklearn.svm.SVC object>,
        'builder': {
            'keywords': [...],
            'keyword_to_idx': {...},
            'alpha': float,
            'alpha_prime': float,
            'f1_score': float
        },
        'meta': {...}
    }
    """
    global MODEL, BUILDER, MODEL_META
    
    print(f"\n[Loading sklearn SVM Model] Path: {SVMODEL_PATH}")
    
    #cek keberadaan model
    if not os.path.exists(SVMODEL_PATH):
        print(f"File tidak ditemukan: {SVMODEL_PATH}")
        print(f"Pastikan upload file skmodel.pkl ke folder: {os.path.dirname(SVMODEL_PATH)}")
        return
    
    try:
        #load model dengan pickle
        with open(SVMODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ File loaded. Type: {type(data)}")
        
        #ekstraksi model dan buildernya
        if isinstance(data, dict):
            #periksa struktur model
            if 'model' not in data:
                print("Key 'model' tidak ada di file!")
                return
            if 'builder' not in data:
                print("Key 'builder' tidak ada di file!")
                return
            
            #ambil model
            MODEL = data['model']
            print(f"MODEL loaded: {type(MODEL).__name__}")
            
            #rekonstruksi model
            builder_data = data['builder']
            BUILDER = _reconstruct_builder_from_dict(builder_data)
            print(f"BUILDER loaded: {type(BUILDER).__name__}")
            
            MODEL_META = data.get('meta', {})
            if MODEL_META:
                print(f"Meta: {json.dumps(MODEL_META)}")
            
            #validasi keberadaan model dan builder
            if MODEL is None:
                print("MODEL is None!")
                return
            if BUILDER is None:
                print("BUILDER is None!")
                return
            
            #periksa isi model
            if not hasattr(MODEL, 'predict'):
                print("Model tidak punya method 'predict'!")
                return
            if not hasattr(BUILDER, 'build_vectors'):
                print("BUILDER tidak punya method 'build_vectors'!")
                return
            
            print(f"Model & Builder Complete!")
            if hasattr(BUILDER, 'all_keywords'):
                print(f"   Keywords: {len(BUILDER.all_keywords)}")
                print(f"   Alpha: {builder_data.get('alpha', 'N/A')}")
                print(f"   Alpha Prime: {builder_data.get('alpha_prime', 'N/A')}")
                if isinstance(builder_data.get('f1_score'), (int, float)):
                    print(f"   F1 Score: {builder_data.get('f1_score'):.4f}")
        
        else:
            print(f"Data bukan dict, type: {type(data)}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


# ===== LOAD SAAT STARTUP =====
_load_sklearn_svm_model()


#metriks numpy
def _confusion_matrix_binary(y_true, y_pred):
    """Return [[TN, FP],[FN, TP]] untuk label biner {0,1}."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)

def _accuracy(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0

def _precision(y_true, y_pred):
    cm = _confusion_matrix_binary(y_true, y_pred)
    tp = cm[1,1]; fp = cm[0,1]
    denom = tp + fp
    return float(tp/denom) if denom > 0 else 0.0

def _recall(y_true, y_pred):
    cm = _confusion_matrix_binary(y_true, y_pred)
    tp = cm[1,1]; fn = cm[1,0]
    denom = tp + fn
    return float(tp/denom) if denom > 0 else 0.0

def _f1(y_true, y_pred):
    p = _precision(y_true, y_pred)
    r = _recall(y_true, y_pred)
    denom = (p + r)
    return float(2*p*r/denom) if denom > 0 else 0.0

def _make_confusion_matrix_from_rating(df_predictions):
    """
    Confusion matrix & metrics: Actual dari rating vs Predicted dari model (kelas biner).
    Mapping rating:
      - 4-5 → 1 (pos)
      - 1-2 → 0 (neg)
      - 3   → ambigu: gunakan ratio 0.5 untuk biner
    """
    df = df_predictions.copy()

    def rating_to_sentiment(row):
        try:
            r = float(row['rating'])
            if r >= 4: return 1
            if r <= 2: return 0
            # rating 3 → ambigu: biner berdasar ratio 0.5
            return 1 if row.get('sentiment_ratio', 0.0) >= 0.5 else 0
        except:
            return 1 if row.get('sentiment_ratio', 0.0) >= 0.5 else 0

    df['actual_binary'] = df.apply(rating_to_sentiment, axis=1).astype(int)

    pred_bin = []
    for _, r in df.iterrows():
        cls = int(r.get('sentiment_class', 1))
        if cls == 2:
            pred_bin.append(1 if r.get('sentiment_ratio', 0.0) >= 0.5 else 0)
        else:
            pred_bin.append(cls)
    df['pred_binary'] = np.array(pred_bin, dtype=int)

    y_actual = df['actual_binary'].values
    y_pred = df['pred_binary'].values

    cm = _confusion_matrix_binary(y_actual, y_pred)
    acc = _accuracy(y_actual, y_pred)
    prec = _precision(y_actual, y_pred)
    rec = _recall(y_actual, y_pred)
    f1  = _f1(y_actual, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual (from Rating)')
    plt.title('Confusion Matrix (Rating vs Model Prediction)')
    cm_b64 = _fig_to_b64img()

    print("CONFUSION MATRIX RESULTS (Rating vs Predicted Sentiment)")
    print(f"Total samples: {len(df)}")
    print(f"\nTest Set Performance (Rating vs Predicted):")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n")

    return cm_b64, {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'total_samples': int(len(df))
    }

def _make_sentiment_dist_img(df_final):
    """Bar chart distribusi sentiment_class (test) -> base64 png."""
    plt.figure(figsize=(8, 5))
    counts = df_final['sentiment_class'].value_counts().sort_index()
    labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
    x_labels = [labels.get(i, f'Class {i}') for i in sorted(df_final['sentiment_class'].unique())]
    plt.bar(x_labels, [counts.get(i, 0) for i in sorted(df_final['sentiment_class'].unique())])
    plt.title('Sentiment Distribution per Review (Test)')
    plt.ylabel('Count')
    return _fig_to_b64img()



def process_sentence_full_optimized(sentence_tuple):
    review_id, review_original, rating, sentence = sentence_tuple

    stemmer = get_stemmer()
    stopwords_id = get_stopwords()  
    kamus_slang = get_kamus_slang()

    text = str(sentence)

    # pembersihan data
    text = _REGEX_HASHTAG.sub(' ', text)
    text = _REGEX_URL.sub(' ', text)
    text = _REGEX_NUMBERS.sub(' ', text)
    text = _REGEX_SPECIAL.sub(" ", text)
    text = _REGEX_SINGLE.sub(" ", text)
    text = _REGEX_MULTI_SPACE.sub(' ', text).strip()

    # penghapusan emoji
    text = text.encode('ascii', 'ignore').decode('ascii')

    # normalisasi karakter berulang
    text = _REGEX_REPEATED.sub(r'\1', text)

    # casefolding dan pemecahan kalimat
    text = text.lower()
    tokens = text.split()

    # proses normalisas slang, stemming, dan penghapusan stopwords
    result_tokens = []
    for word in tokens:
        if not word or not word[0].isalpha():
            continue
        word = kamus_slang.get(word, word) #normalisasi slang
        word = stemmer.stem(word) #stemming
        if word not in stopwords_id: #stopword removal
            result_tokens.append(word)

    tokens_str = ' '.join(result_tokens)

    if tokens_str.strip():
        return {
            'review_id': review_id,
            'review_original': review_original,
            'rating': rating,
            'sentence': sentence,
            'tokens_str': tokens_str,
            'tokens': result_tokens
        }
    else:
        return None

def preprocess_pipeline_fast_v2(ulasan_df, n_workers=None):
    
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)



    #menyiapkan data
    ulasan = ulasan_df[['Review', 'Rating']].copy()
    ulasan.columns = ['ulasan', 'Rating']

    ulasan = ulasan.dropna(subset=['ulasan'])
    ulasan = ulasan[ulasan['ulasan'].astype(str).str.strip() != '']
    ulasan = ulasan.reset_index(drop=True)

    #membuat id untuk setiap review
    ulasan['review_id'] = range(1, len(ulasan) + 1)

    #pemecahan kalimat
    rows_to_process = []

    for idx, row in ulasan.iterrows():
        if idx % 5000 == 0 and idx > 0:
            print(f"      {idx}/{len(ulasan)} reviews...")

        review_id = row['review_id']
        review_text = str(row['ulasan'])
        rating = row['Rating']
        #penghapusan karakter non ASCII
        review_text_clean = review_text.encode('ascii', 'ignore').decode('ascii')
        sentences = _REGEX_SPLIT_SENTENCE.split(review_text_clean)

        for sentence in sentences:
            s = sentence.strip()
            if s and len(s) > 2:
                rows_to_process.append((review_id, review_text, rating, s))

    print(f"      Total kalimat: {len(rows_to_process)}")

    #processing
    start_time = time.time()

    with Pool(processes=n_workers) as pool:
        chunksize = max(100, len(rows_to_process) // (n_workers * 4))
        results = pool.map(
            process_sentence_full_optimized,
            rows_to_process,
            chunksize=chunksize
        )

    results = [r for r in results if r is not None]

    elapsed = time.time() - start_time
    processed_per_sec = len(rows_to_process) / elapsed if elapsed > 0 else 0

    print(f"      {len(results)} sentences after cleanup")

    #membuat dataframe
    if results:
        ulasan_final = pd.DataFrame(results)
        ulasan_final = ulasan_final[[
            'review_id', 'review_original', 'rating', 'sentence', 'tokens_str', 'tokens'
        ]]
    else:
        ulasan_final = pd.DataFrame()

    if len(ulasan_final) > 0:
        print(ulasan_final[['review_id', 'rating', 'sentence']].head(10))
    else:
        print("No result")

    return ulasan_final

#other navigations
def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def help(request):
    return render(request, 'help.html')

#untuk download template
def download_template(request):
    template_type = request.GET.get('download_template')
    file_format = request.GET.get('format', 'csv')

    if template_type not in ['test'] or file_format not in ['csv', 'xlsx']:
        return HttpResponse('Invalid parameters', status=400)

    try:
        if template_type == 'train':
            filename = f'template_latih.{file_format}'
        else:
            filename = f'template_uji.{file_format}'

        template_path = os.path.join(settings.BASE_DIR, filename)

        if not os.path.exists(template_path):
            return HttpResponse(f'Template file not found: {filename}', status=404)

        with open(template_path, 'rb') as f:
            file_content = f.read()

        if file_format == 'csv':
            content_type = 'text/csv'
        else:
            content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'

        response = HttpResponse(file_content, content_type=content_type)
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    except Exception as e:
        return HttpResponse(f'Error: {str(e)}', status=500)

#halaman preprocess
def preprocess(request):
    if request.method == 'GET':
        download_type = request.GET.get('download')
        if download_type:
            return handle_download(request, download_type)
        return render(request, 'preprocess.html')
    elif request.method == 'POST':
        action = request.POST.get('action')
        if action == 'preview':
            return handle_preview(request)
        elif action == 'process':
            return handle_process(request)
    return render(request, 'preprocess.html')

def handle_preview(request):
    try:
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'success': False, 'error': 'No file uploaded'})

        df = _read_any(file)

        if 'Review' not in df.columns or 'Rating' not in df.columns:
            return JsonResponse({
                'success': False,
                'error': 'File harus memiliki kolom "Review" dan "Rating"'
            })

        preview_data = df.head(10).to_dict('records')
        return JsonResponse({
            'success': True,
            'total_rows': len(df),
            'columns': df.columns.tolist(),
            'preview': preview_data
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def handle_process(request):
    try:
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'success': False, 'error': 'No file uploaded'})

        df = _read_any(file)

        if 'Review' not in df.columns or 'Rating' not in df.columns:
            return JsonResponse({
                'success': False,
                'error': 'File harus memiliki kolom "Review" dan "Rating"'
            })

        #proses lebih cepat
        ulasan_final = preprocess_pipeline_fast_v2(df)

        if len(ulasan_final) == 0:
            return JsonResponse({
                'success': False,
                'error': 'Tidak ada data yang tersisa setelah preprocessing'
            })

        #hitung hasil statistik
        total_reviews = len(set(ulasan_final['review_id']))
        total_sentences = len(ulasan_final)
        avg_sentences = total_sentences / max(total_reviews, 1)

        print(f"\nPreprocessing Done!")
        print(f"Statistik:")
        print(f"  Total review: {total_reviews}")
        print(f"  Total kalimat: {total_sentences}")
        print(f"  Rata-rata kalimat per review: {avg_sentences:.2f}")

        #simpan hasil agar bisa diunduh
        request.session['processed_data'] = ulasan_final.to_json(orient='records')
        request.session['processed_readable'] = ulasan_final[[
            'review_id', 'rating', 'sentence', 'tokens_str'
        ]].to_json(orient='records')

        # Siapkan semua data untuk pagination (tidak hanya 10 baris)
        display_df = ulasan_final[[
            'review_id', 'rating', 'sentence', 'tokens_str'
        ]]
        all_preview_data = display_df.to_dict('records')

        return JsonResponse({
            'success': True,
            'stats': {
                'total_reviews': total_reviews,
                'total_sentences': total_sentences,
                'avg_sentences': f"{avg_sentences:.2f}"
            },
            'preview': all_preview_data,  # Kirim SEMUA data, bukan cuma 10
            'total_rows': len(all_preview_data)
        })

    except Exception as e:
        import traceback
        return JsonResponse({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

def handle_download(request, download_type):
    try:
        if download_type == 'full':
            data_json = request.session.get('processed_data')
            filename = 'data_bersih_final.csv'
        elif download_type == 'readable':
            data_json = request.session.get('processed_readable')
            filename = 'data_bersih_readable.csv'
        else:
            return HttpResponse('Invalid download type', status=400)

        if not data_json:
            return HttpResponse('No data to download', status=400)

        df = pd.read_json(data_json, orient='records')
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        df.to_csv(response, index=False, encoding='utf-8-sig')
        return response
    except Exception as e:
        return HttpResponse(f'Error: {str(e)}', status=500)


#halaman sentiment analysis
def sentiment(request):
    if request.method == 'GET':
        if request.GET.get('download_template'):
            return download_template(request)

        dl = request.GET.get('download')
        if dl == 'results':
            data_json = request.session.get('sent_results_json')
            if not data_json:
                return HttpResponse('No results available', status=400)
            df = pd.read_json(data_json, orient='records')
            resp = HttpResponse(content_type='text/csv')
            resp['Content-Disposition'] = 'attachment; filename="sentiment_analysis_results.csv"'
            df.to_csv(resp, index=False, encoding='utf-8-sig')
            return resp

        if dl == 'report':
            report_text = request.session.get('sent_report_text')
            if not report_text:
                return HttpResponse('No report available', status=400)
            resp = HttpResponse(report_text, content_type='text/plain; charset=utf-8')
            resp['Content-Disposition'] = 'attachment; filename="sentiment_report.txt"'
            return resp

        return render(request, 'sentiment.html')

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'preview':
            try:
                file = request.FILES.get('file')
                if not file:
                    return JsonResponse({'success': False, 'error': 'No file uploaded'})
                df = _read_any(file)
                
                #validasi jumlah baris agar tidak kurang dari minimal (100)
                
                
                return JsonResponse({
                    'success': True,
                    'rows': int(len(df)),
                    'total_rows': int(len(df))
                })
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)})

        if action == 'analyze':
            try:
                #test file untuk sentiment analysis
                test_file = request.FILES.get('test_file')
                if not test_file:
                    return JsonResponse({
                        'success': False,
                        'error': 'Missing test_file'
                    })

                df_test = _read_any(test_file)
                
             

                #validasi adanya model/tidak
                if MODEL is None or BUILDER is None:
                    return JsonResponse({
                        'success': False,
                        'error': 'Model belum tersedia. Pastikan sentiment_app/skmodel.pkl berisi {model,builder}.'
                    })

                required_cols = ['sentence', 'review_original', 'rating', 'review_id']
                for col in required_cols:
                    if col not in df_test.columns:
                        return JsonResponse({
                            'success': False,
                            'error': f'Data test harus memiliki kolom \"{col}\"'
                        })

                df_test = df_test.dropna(subset=['sentence']).copy()

                #prediksi sentiment per kalimat
                X_test = df_test['sentence'].astype(str).values
                Xv_test = BUILDER.build_vectors(X_test)
                y_pred_test = np.asarray(MODEL.predict(Xv_test)).astype(int)

                #simpan prediksi sentiment per kalimat, diagregasi per review
                df_pred = df_test[['review_id', 'review_original', 'rating']].copy()
                df_pred['sentiment'] = y_pred_test

                agg = df_pred.groupby(['review_id']).agg(
                    review_original=('review_original', 'first'),
                    rating=('rating', 'first'),
                    positive_count=('sentiment', 'sum'),
                    total_sentences=('sentiment', 'size')
                ).reset_index()

                agg['sentiment_ratio'] = agg['positive_count'] / agg['total_sentences'].replace(0, np.nan)
                agg['sentiment_ratio'] = agg['sentiment_ratio'].fillna(0.0)

                def _classify_ratio(r, lower=0.4, upper=0.6):
                    if r < lower:  return 0
                    if r > upper:  return 1
                    return 2

                agg['sentiment_class'] = agg['sentiment_ratio'].apply(_classify_ratio).astype(int)

                #confusion matrix
                cm_b64, test_metrics = _make_confusion_matrix_from_rating(agg)

                #distribusi sentiment
                dist_b64 = _make_sentiment_dist_img(agg)

                #korelasi sentiment rasio dan rating bintang
                try:
                    rating_num = pd.to_numeric(agg['rating'], errors='coerce')
                    sr = pd.Series(agg['sentiment_ratio'])
                    pearson = float(sr.corr(rating_num, method='pearson'))
                    spearman = float(sr.corr(rating_num, method='spearman'))
                    if np.isnan(pearson): pearson = 0.0
                    if np.isnan(spearman): spearman = 0.0
                except Exception:
                    pearson = 0.0
                    spearman = 0.0

                #preview hasil sentiment analysis 15 baris pertama
                sample = agg.to_dict(orient='records')

                #export csv
                def _rating_to_original(r):
                    try:
                        rr = float(r)
                        if rr >= 4: return 1
                        if rr <= 2: return 0
                        return 2
                    except:
                        return 2

                export_df = agg.copy()
                export_df['sentiment_asli'] = export_df['rating'].apply(_rating_to_original)

                export_df = export_df.rename(columns={
                    'review_id': 'review_id',
                    'review_original': 'review_original',
                    'rating': 'rating',
                    'total_sentences': 'jumlah_kalimat_total',
                    'positive_count': 'jumlah_kalimat_positif',
                    'sentiment_ratio': 'rasio_sentiment',
                    'sentiment_class': 'sentiment_prediksi'
                })

                export_df = export_df[[
                    'review_id',
                    'review_original',
                    'rating',
                    'jumlah_kalimat_total',
                    'jumlah_kalimat_positif',
                    'rasio_sentiment',
                    'sentiment_prediksi',
                    'sentiment_asli'
                ]]

                request.session['sent_results_json'] = export_df.to_json(orient='records')

                # metrik evaluasi model 
                cv_metrics = {
                    'accuracy': float(test_metrics['accuracy']),
                    'precision': float(test_metrics['precision']),
                    'recall': float(test_metrics['recall']),
                    'f1_score': float(test_metrics['f1_score'])
                }
                folds = []

                #laporan ringkasan
                report_text = []
                report_text.append("SENTIMENT ANALYSIS REPORT (NO-TRAIN MODE)")
                report_text.append(f"    Test rows (sentences): {len(df_test)}")
                report_text.append(f"    Test reviews: {agg['review_id'].nunique()}")
                report_text.append(f"    Loaded from: {SVMODEL_PATH}")
                if MODEL_META:
                    report_text.append(f"    Meta: {json.dumps(MODEL_META)}")
                report_text.append(f"    Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
                report_text.append(f"    Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
                report_text.append(f"    Recall:    {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
                report_text.append(f"    F1-Score:  {test_metrics['f1_score']:.4f} ({test_metrics['f1_score']*100:.2f}%)")
                report_text.append(f"    Pearson r:  {pearson:.4f}")
                report_text.append(f"    Spearman ρ: {spearman:.4f}")
                request.session['sent_report_text'] = "\n".join(report_text)

                return JsonResponse({
                    'success': True,
                    'cv_metrics': cv_metrics,
                    'test_metrics': test_metrics,
                    'cv_results': folds,
                    'confusion_matrix_img': cm_b64,
                    'sentiment_dist_img': dist_b64,
                    'correlation': pearson,
                    'correlation_pearson': pearson,
                    'correlation_spearman': spearman,
                    'sample_predictions': sample,
                    'log': "Analyze done\n"
                })
            except Exception as e:
                import traceback
                return JsonResponse({
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })

    return HttpResponse(status=405)