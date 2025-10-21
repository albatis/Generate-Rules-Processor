import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
#from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pickle
import os
import time

def main():
    input_file_path = os.getenv('INPUT_FILE_PATH', '/mnt/2023_spotify_ds1.csv')
    output_file_path = os.getenv('OUTPUT_MODEL_PATH', '/mnt/association_rules.pkl')
    try:
        min_support = float(os.getenv('MIN_SUPPORT', '0.01'))
        min_confidence = float(os.getenv('MIN_CONFIDENCE', '0.2'))
    except ValueError:
        print("As variáveis de ambiente MIN_SUPPORT ou MIN_CONFIDENCE estão com valores inválidos por isso serão utilizados os valores default")
        min_support = 0.01
        min_confidence = 0.2
    try:
        df = pd.read_csv(input_file_path)
        df_playlists = df[['pid', 'track_name']]
        print(f"O arquivo contém {df_playlists.shape[0]} registros encontrados")
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{input_file_path}' não foi encontrado")
        return

    start = time.time()
    playlists = df_playlists.drop_duplicates(subset=['pid', 'track_name']).groupby('pid')['track_name'].apply(list).tolist()
    end = time.time()
    print(f"{len(playlists)} playlists criadas: [{end - start:.2f} s]")

    start = time.time()
    te = TransactionEncoder()
    te_ary = te.fit(playlists).transform(playlists)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    end = time.time()
    print(f"Encoded: [{end - start:.2f} s]")

    start = time.time()
    #frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    end = time.time()
    print(f" {len(frequent_itemsets)} conjuntos de itens frequentes: [{end - start:.2f} s]")

    if frequent_itemsets.empty:
        print("Nenhum conjunto de itens frequentes encontrado. Tente um suporte menor")
        return

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules['lift'] > 1.5]
    print(f"{len(rules)} regras de associação geradas")

    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(rules, f)
        print(f"Modelo salvo em: ['{output_file_path}']")
    except Exception as e:
        print(f"ERRO: não foi possível salvar o arquivo do modelo: {e}")
    print(" Fim do processamento ")

if __name__ == '__main__':
    main()