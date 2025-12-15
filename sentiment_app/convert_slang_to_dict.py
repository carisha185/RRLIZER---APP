"""
Script untuk convert kamus slang dari CSV ke Python dictionary
Jalankan sekali saja untuk generate file slang_dictionary.py
"""

import pandas as pd

kamus_slang_df = pd.read_csv('full.csv')


slang_col = 'transformed'  #kolom untuk kata slang
formal_col = 'original-for'  #kolom untuk kata formal/asli

#konversi ke dictionary / kamus
kamus_dict = dict(zip(kamus_slang_df[slang_col], kamus_slang_df[formal_col]))

# bikin file python untuk simpan kamus
with open('sentiment_app/slang_dictionary.py', 'w', encoding='utf-8') as f:
    f.write('# sentiment_app/slang_dictionary.py\n')
    f.write('# Kamus normalisasi slang bahasa Indonesia (Auto-generated)\n')
    f.write('# Sumber: IndoCollex\n\n')
    f.write('KAMUS_SLANG = {\n')
    
    #tulis setiap entry
    for slang, formal in kamus_dict.items():
        #kalau data punya kutip, ditambah backslash sebagai penanda biar ga error
        slang_safe = str(slang).replace("'", "\\'")
        formal_safe = str(formal).replace("'", "\\'")
        f.write(f"    '{slang_safe}': '{formal_safe}',\n")
    
    f.write('}\n')

print(f"Kamus berhasil dibuat!")
