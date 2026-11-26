# NBA Props Analyzer 

## Como rodar ?
1. Crie um ambiente virtual novo.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Copie `.streamlit/secrets.example.toml` para `.streamlit/secrets.toml`.
4. Coloque sua chave da Odds API em `.streamlit/secrets.toml`.
5. Rode o app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Mudanças
- Visão **Prop Bet Cheat Sheet** (tabela principal estilo BettingPros).
- **Rating 1–5 estrelas**, projeção vs linha, L5/L10 e filtros por jogo/lado/cover prob.
- Visões **Cheat Sheet**, **Top picks** e **Por jogo**.

