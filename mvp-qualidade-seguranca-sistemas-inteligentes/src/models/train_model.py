import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import logging
import os # Importar os para manipulação de caminhos

def load_processed_data(filepath):
    """
    Carrega o dataset pré-processado.
    """
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Dataset processado carregado com sucesso de: {filepath}")
        return df
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo não foi encontrado em {filepath}")
        return None
    except Exception as e:
        logging.error(f"Erro ao carregar o dataset processado: {e}")
        return None

def train_model(df, target_column='Addicted_Score'):
    """
    Treina um modelo de machine learning.
    Neste exemplo, usaremos RandomForestClassifier.
    """
    if df is None:
        logging.error("Não há dados para treinar o modelo. O DataFrame de entrada é None.")
        return None, None, None, None, None

    if target_column not in df.columns:
        logging.error(f"Coluna alvo '{target_column}' não encontrada no dataset. Por favor, verifique o nome da coluna alvo.")
        return None, None, None, None, None

    # As features (X) serão todas as outras colunas exceto a coluna alvo
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- TRECHO PARA TRATAR CLASSES RARAS ---
    class_counts = y.value_counts()
    rare_classes = class_counts[class_counts < 2].index.tolist() # Converter para lista para logging

    if rare_classes:
        logging.warning(f"As seguintes classes na coluna alvo '{target_column}' têm menos de 2 membros e serão removidas antes do split: {rare_classes}")
        initial_rows = len(y)
        X = X[~y.isin(rare_classes)]
        y = y[~y.isin(rare_classes)]
        logging.warning(f"Removidas {initial_rows - len(y)} linhas devido a classes raras na coluna alvo.")

        if y.empty: # Caso o dataset alvo fique vazio após a remoção
            logging.error("O dataset alvo ficou vazio após a remoção de classes raras. Não é possível treinar o modelo.")
            return None, None, None, None, None
    # --- FIM DO TRECHO DE TRATAMENTO DE CLASSES RARAS ---

    logging.info(f"Dividindo os dados em conjuntos de treino e teste para a coluna alvo '{target_column}'...")

    # Verificar se 'y' tem mais de uma classe para estratificação após a remoção de raras
    if y.nunique() > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        logging.warning(f"A coluna alvo '{target_column}' tem apenas uma classe após o tratamento de raras, ou poucas para estratificação. Não será possível usar stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if y.nunique() == 1:
             logging.error(f"A coluna alvo '{target_column}' tem apenas uma classe ({y.iloc[0]}) após a divisão. O treinamento de um classificador pode não ser significativo.")


    logging.info("Iniciando o treinamento do modelo RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Modelo treinado com sucesso.")

    logging.info("Avaliando o modelo no conjunto de teste...")
    y_pred = model.predict(X_test)

    if y_test.nunique() > 1:
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        logging.info(f"Acurácia do modelo: {accuracy:.4f}")
        logging.info(f"Relatório de Classificação:\n{report}")
    else:
        logging.warning(f"Não foi possível gerar um relatório de classificação completo: o conjunto de teste para '{target_column}' contém apenas uma classe.")
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Acurácia do modelo (simplificada): {accuracy:.4f}")

    return model, X_test, y_test, y_pred, accuracy

def save_model(model, filepath):
    """
    Salva o modelo treinado.
    """
    if model is None:
        logging.error("Não há modelo para salvar. O modelo de entrada é None.")
        return

    try:
        # Garante que o diretório do modelo exista
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        logging.info(f"Modelo salvo com sucesso em: {filepath}")
    except Exception as e:
        logging.error(f"Erro ao salvar o modelo: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Caminho base do projeto
    # ATENÇÃO: Ajuste este caminho para a *raiz* do seu repositório.
    # Esta é a pasta 'mvp-qualidade-seguranca-sistemas-inteligentes' que contém 'data', 'models' e 'src'.
    project_root = 'C:/Users/leonardo.romariz/Documents/mvp-qualidade-seguranca-sistemas-inteligentes/mvp-qualidade-seguranca-sistemas-inteligentes'

    # Caminho do dataset processado
    # DEVE APONTAR PARA A PASTA 'data/processed'
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'students_social_media_addiction_processed.csv')

    # Caminho para salvar o modelo treinado
    # DEVE APONTAR PARA A PASTA 'models'
    model_path = os.path.join(project_root, 'models', 'random_forest_addiction_model.joblib')

    logging.info("Iniciando o pipeline de treinamento do modelo...")

    # 1. Carregar dados processados
    data = load_processed_data(processed_data_path)

    if data is not None:
        # Coluna alvo agora é explicitamente 'Addicted_Score'
        target_col = 'Addicted_Score'

        if target_col not in data.columns:
            logging.error(f"Coluna alvo '{target_col}' não encontrada no dataset. Certifique-se de que o arquivo CSV original contém esta coluna e que o make_dataset.py foi executado corretamente.")
            exit()

        model, X_test, y_test, y_pred, accuracy = train_model(data, target_column=target_col)

        if model is not None:
            # 3. Salvar o modelo treinado
            save_model(model, model_path)
        else:
            logging.error("Treinamento do modelo falhou.")
    else:
        logging.error("Carregamento de dados processados falhou.")

    logging.info("Pipeline de treinamento do modelo concluído.")