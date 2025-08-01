# 🎥 Sistema de Recomendação Multiplataforma de Filmes e Séries

Este projeto desenvolve um **sistema de recomendação híbrido** que integra os catálogos de **Netflix, Amazon Prime Video e Disney+** em um único *data lake*. A proposta é oferecer ao usuário sugestões personalizadas com base em:

## 🧠 Critérios de Recomendação

1. **Popularidade Temporal**

   - Títulos lançados nos **últimos dois anos** recebem maior pontuação, com **decaimento exponencial** para obras mais antigas.

2. **Disponibilidade de Plataforma**

   - O sistema exibe **apenas recomendações das plataformas que o usuário realmente assina**, evitando sugestões inacessíveis.

---

## 🛠️ Tecnologia e Funcionalidades

- Unificação de catálogos via `pandas`.
- Cálculo de score de popularidade temporal com base no ano de lançamento.
- Interface interativa desenvolvida com **Streamlit**, oferecendo:
  - Abas por plataforma: **Netflix | Prime | Disney+**
  - Recomendação dinâmica com base nas plataformas selecionadas
  - Exibição dos **10 títulos mais populares** em tempo real por aba

---

## 📁 Arquitetura

- **Entrada**:

  - `netflix_titles.csv`
  - `amazon_prime_titles.csv`
  - `disney_plus_titles.csv`

- **Processamento**:

  - Unificação dos dados
  - Cálculo do score de popularidade
  - Filtragem por plataforma assinada

- **Saída**:

  - Aplicativo interativo com recomendações personalizadas

---

## ▶️ Como Executar

1. Instale as dependências (caso necessário):

   ```bash
   pip install streamlit pandas sklearn
   ```

2. Coloque o arquivo `unified_catalog.csv` na mesma pasta do script.

3. Execute a aplicação:

   ```bash
   streamlit run recomendador_streamlit.py
   ```

