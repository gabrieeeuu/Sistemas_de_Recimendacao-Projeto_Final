from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np
import pandas as pd

# Caminhos dos arquivos CSV
netflix_path = "./datasets/netflix_titles.csv"
amazon_path = "./datasets/amazon_prime_titles.csv"
disney_path = "./datasets/disney_plus_titles.csv"

# Carregando os arquivos
df_netflix = pd.read_csv(netflix_path)
df_amazon = pd.read_csv(amazon_path)
df_disney = pd.read_csv(disney_path)

# Visualizando o início de cada DataFrame
st.header("Amostra dos Dados - Netflix")
st.dataframe(df_netflix.head())
st.header("Amostra dos Dados - Amazon Prime")
st.dataframe(df_amazon.head())
st.header("Amostra dos Dados - Disney+")
st.dataframe(df_disney.head())

df_netflix['platform'] = 'Netflix'
df_amazon['platform'] = 'Amazon Prime'
df_disney['platform'] = 'Disney+'

# Padronizando os nomes das colunas
common_columns = ['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 
                  'release_year', 'rating', 'duration', 'listed_in', 'description', 'platform']

# Unificando os datasets em um único DataFrame
df_merged = pd.concat([df_netflix[common_columns], 
                       df_amazon[common_columns], 
                       df_disney[common_columns]], ignore_index=True)

# Pré-processamento: combinando campos para vetorização de conteúdo
def combine_features(row):
    return ' '.join(str(val) for val in [row['listed_in'], row['description'], row['cast'], row['director']] if pd.notnull(val))

df_merged['combined_features'] = df_merged.apply(combine_features, axis=1)

# Vetorização TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_merged['combined_features'])

# Similaridade de conteúdo
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Popularidade temporal: score baseado em recência (últimos 2 anos com decaimento exponencial)
CURRENT_YEAR = 2025

def popularity_score(year):
    age = CURRENT_YEAR - year
    return np.exp(-0.5 * age) if age >= 0 else 0

df_merged['popularity_score'] = df_merged['release_year'].apply(popularity_score)

# Função para recomendar Top-N híbrido
def get_hybrid_recommendations(title, platform_filter, top_n=10, alpha=0.7):
    idx = df_merged[df_merged['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return pd.DataFrame()
    
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Combinação com score de popularidade
    scores = []
    for i, score in sim_scores:
        content_score = score
        popularity = df_merged.iloc[i]['popularity_score']
        hybrid = alpha * content_score + (1 - alpha) * popularity
        scores.append((i, hybrid))

    # Filtra por plataforma
    scores = [s for s in scores if df_merged.iloc[s[0]]['platform'] in platform_filter]
    
    # Ordena pelo score híbrido
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Coleta os top N (excluindo o próprio título)
    top_indices = [i for i, _ in scores if i != idx][:top_n]
    
    return df_merged.iloc[top_indices][['title', 'platform', 'release_year', 'listed_in', 'description']]

# Caminho do arquivo unificado
unified_path = "./datasets/unified_catalog.csv"

# Normalizando o score para valores entre 0 e 1
df_merged['popularity_score'] = (
    (df_merged['popularity_score'] - df_merged['popularity_score'].min()) /
    (df_merged['popularity_score'].max() - df_merged['popularity_score'].min())
)

# Salvando base e função para uso no Streamlit
df_merged.to_csv(unified_path, index=False)

# Visualizando os títulos mais populares segundo o score temporal
st.header("Score de Popularidade")
top_popular = df_merged.sort_values(by='popularity_score', ascending=False).head(10)
top_popular[['title', 'platform', 'release_year', 'popularity_score']]

st.title("🎬 Recomendador de Filmes e Séries por Plataforma")

# Plataformas disponíveis
platforms = ['Netflix', 'Amazon Prime', 'Disney+']

st.markdown("---")

# Filtro de gêneros (listed_in)
all_genres = set()
df_merged['listed_in'].dropna().apply(lambda x: [all_genres.add(g.strip()) for g in x.split(',')])
selected_genres = st.multiselect("Filtrar por Gênero", sorted(all_genres))

if selected_genres:
  genre_mask = df_merged['listed_in'].apply(
    lambda x: any(g in x for g in selected_genres) if pd.notnull(x) else False
  )
  df_merged = df_merged[genre_mask]

# Número de recomendações por aba
top_n = 10

# Função para mostrar recomendações por plataforma
def show_recommendations(platform_name):
    st.subheader(f"Recomendações - {platform_name}")
    filtered = df_merged[(df_merged['platform'] == platform_name) & (df_merged['platform'].isin(platforms))]
    top_items = filtered.sort_values(by='popularity_score', ascending=False).head(top_n)
    for _, row in top_items.iterrows():
        with st.container():
            st.markdown(f"### {row['title']} ({int(row['release_year'])})")
            st.markdown(f"**Gêneros:** {row['listed_in']}")
            st.markdown(f"**Descrição:** {row['description']}")
            st.markdown("---")

# Abas por plataforma
tab1, tab2, tab3 = st.tabs(platforms)

with tab1:
    show_recommendations("Netflix")

with tab2:
    show_recommendations("Amazon Prime")

with tab3:
    show_recommendations("Disney+")
