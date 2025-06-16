# Arquivo: app.py (Versão Final e Completa)
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Previsor de Partidas", page_icon="⚽", layout="wide")

# --- FUNÇÃO PARA CARREGAR OS DADOS (cache para performance) ---
@st.cache_data
def carregar_dados():
    """Carrega o modelo, as colunas e o dataset com features, uma única vez."""
    try:
        model = joblib.load('modelo_futebol.pkl')
        model_columns = joblib.load('model_columns.pkl')
        df_features = pd.read_csv('dataset_com_features.csv')
        df_features['Date'] = pd.to_datetime(df_features['Date'])
        lista_times = sorted(list(set([col.split('_')[1] for col in model_columns if '_' in col])))
        return model, model_columns, df_features, lista_times
    except FileNotFoundError:
        return None, None, None, None

# Carrega os dados na inicialização
model, model_columns, df_features, lista_times = carregar_dados()

if model is None:
    st.error("ERRO CRÍTICO: Arquivos de modelo ou de features não encontrados. Execute toda a esteira de dados e treinamento.")
    st.stop()

# --- INTERFACE DO USUÁRIO ---
st.title('🤖 Previsor de Partidas de Futebol')
st.markdown("### Previsões para o Mundial de Clubes em Campo Neutro")

col1, col2 = st.columns(2)
with col1:
    time_A = st.selectbox('Selecione o Time A', lista_times, index=lista_times.index('Al Ahly') if 'Al Ahly' in lista_times else 0)
with col2:
    time_B = st.selectbox('Selecione o Time B', lista_times, index=lista_times.index('Inter Miami') if 'Inter Miami' in lista_times else 1)

if st.button('Fazer Previsão', type="primary", use_container_width=True):
    if time_A == time_B:
        st.error("Os times A e B devem ser diferentes.")
    else:
        try:
            # --- LÓGICA AUTOMÁTICA DE BUSCA DE FEATURES ---
            stats_time_A = df_features[(df_features['HomeTeam'] == time_A) | (df_features['AwayTeam'] == time_A)].sort_values(by='Date').iloc[-1]
            stats_time_B = df_features[(df_features['HomeTeam'] == time_B) | (df_features['AwayTeam'] == time_B)].sort_values(by='Date').iloc[-1]

            if stats_time_A['HomeTeam'] == time_A:
                forma_A, gf_A, gs_A = stats_time_A[['Forma_Casa_5J', 'Media_GF_Casa_5J', 'Media_GS_Casa_5J']]
            else:
                forma_A, gf_A, gs_A = stats_time_A[['Forma_Visitante_5J', 'Media_GF_Visitante_5J', 'Media_GS_Visitante_5J']]

            if stats_time_B['HomeTeam'] == time_B:
                forma_B, gf_B, gs_B = stats_time_B[['Forma_Casa_5J', 'Media_GF_Casa_5J', 'Media_GS_Casa_5J']]
            else:
                forma_B, gf_B, gs_B = stats_time_B[['Forma_Visitante_5J', 'Media_GF_Visitante_5J', 'Media_GS_Visitante_5J']]

            # --- EXIBIÇÃO DAS ESTATÍSTICAS (INFORMATIVO) ---
            st.markdown("---")
            st.subheader("Estatísticas Recentes (Automáticas)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{time_A}**")
                st.metric(label="Média de Pontos (últimos 5 jogos)", value=f"{forma_A:.2f}")
                st.metric(label="Média de Gols Feitos", value=f"{gf_A:.2f}")
                st.metric(label="Média de Gols Sofridos", value=f"{gs_A:.2f}")
            with c2:
                st.markdown(f"**{time_B}**")
                st.metric(label="Média de Pontos (últimos 5 jogos)", value=f"{forma_B:.2f}")
                st.metric(label="Média de Gols Feitos", value=f"{gf_B:.2f}")
                st.metric(label="Média de Gols Sofridos", value=f"{gs_B:.2f}")
            
            # --- LÓGICA DE PREVISÃO PARA CAMPO NEUTRO ---
            # PREVISÃO 1: Time A como "Casa"
            dados_input_1 = pd.DataFrame(columns=model_columns, index=[0]).fillna(0)
            dados_input_1.loc[0, ['Forma_Casa_5J', 'Media_GF_Casa_5J', 'Media_GS_Casa_5J']] = [forma_A, gf_A, gs_A]
            dados_input_1.loc[0, ['Forma_Visitante_5J', 'Media_GF_Visitante_5J', 'Media_GS_Visitante_5J']] = [forma_B, gf_B, gs_B]
            if 'HomeTeam_' + time_A in dados_input_1.columns: dados_input_1.loc[0, 'HomeTeam_' + time_A] = 1
            if 'AwayTeam_' + time_B in dados_input_1.columns: dados_input_1.loc[0, 'AwayTeam_' + time_B] = 1
            probabilidades_1 = model.predict_proba(dados_input_1)[0]
            
            # PREVISÃO 2: Time B como "Casa"
            dados_input_2 = pd.DataFrame(columns=model_columns, index=[0]).fillna(0)
            dados_input_2.loc[0, ['Forma_Casa_5J', 'Media_GF_Casa_5J', 'Media_GS_Casa_5J']] = [forma_B, gf_B, gs_B]
            dados_input_2.loc[0, ['Forma_Visitante_5J', 'Media_GF_Visitante_5J', 'Media_GS_Visitante_5J']] = [forma_A, gf_A, gs_A]
            if 'HomeTeam_' + time_B in dados_input_2.columns: dados_input_2.loc[0, 'HomeTeam_' + time_B] = 1
            if 'AwayTeam_' + time_A in dados_input_2.columns: dados_input_2.loc[0, 'AwayTeam_' + time_A] = 1
            probabilidades_2 = model.predict_proba(dados_input_2)[0]

            # MÉDIA DAS PROBABILIDADES
            mapa_classes = {classe: i for i, classe in enumerate(model.classes_)}
            prob_A_vence = np.mean([probabilidades_1[mapa_classes.get('H', 0)], probabilidades_2[mapa_classes.get('A', 1)]])
            prob_B_vence = np.mean([probabilidades_1[mapa_classes.get('A', 1)], probabilidades_2[mapa_classes.get('H', 0)]])
            prob_empate = np.mean([probabilidades_1[mapa_classes.get('D', 2)], probabilidades_2[mapa_classes.get('D', 2)]])

            # Determina o resultado final com base na maior probabilidade
            if prob_A_vence > prob_B_vence and prob_A_vence > prob_empate:
                resultado_final = f"Vitória do {time_A}"
            elif prob_B_vence > prob_A_vence and prob_B_vence > prob_empate:
                resultado_final = f"Vitória do {time_B}"
            else:
                resultado_final = "Empate"

            # --- PARTE CORRIGIDA: EXIBIÇÃO DOS RESULTADOS FINAIS ---
            st.markdown("---")
            st.subheader(f'Previsão para: {time_A} vs {time_B} (Campo Neutro)')
            
            if "Vitória" in resultado_final:
                st.success(f'Resultado Mais Provável: {resultado_final}!')
            else:
                st.warning(f'Resultado Mais Provável: {resultado_final}!')
                
            st.write("---")
            st.subheader('Probabilidades Neutras Calculadas:')
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(label=f"Vitória {time_A}", value=f"{prob_A_vence:.2%}")
            with c2:
                st.metric(label="Empate", value=f"{prob_empate:.2%}")
            with c3:
                st.metric(label=f"Vitória {time_B}", value=f"{prob_B_vence:.2%}")

        except IndexError:
            st.error(f"Não foi possível encontrar dados de jogos recentes para um dos times selecionados. O time pode ser novo no dataset ou não ter dados suficientes.")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado durante a previsão: {e}")