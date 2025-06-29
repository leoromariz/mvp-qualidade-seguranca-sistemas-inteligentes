# mvp-qualidade-seguranca-sistemas-inteligentes/src/data/make_dataset.py

import pandas as pd
import logging
import os

# Definição dos mapeamentos manuais para as colunas categóricas
# ATENÇÃO: VERIFIQUE ESTES MAPEAMENTOS COM SEUS DADOS REAIS
# E AJUSTE SE HOUVER OUTRAS CATEGORIAS OU SE QUISER UMA ORDEM ESPECÍFICA.
CATEGORICAL_MAPPINGS = {
    'Gender': {
        'Male': 0, 'Female': 1, 'Other': 2
    },
    'Academic_Level': {
        'Undergraduate': 0, 'Graduate': 1, 'High School': 2, 'Other': 3
    },
    'Country': {
        'Bangladesh': 0, 'India': 1, 'USA': 2, 'UK': 3, 'Canada': 4, 'Australia': 5,
        'Germany': 6, 'Brazil': 7, 'Japan': 8, 'South Korea': 9, 'France': 10,
        'Spain': 11, 'Italy': 12, 'Mexico': 13, 'Russia': 14, 'China': 15,
        'Sweden': 16, 'Norway': 17, 'Denmark': 18, 'Netherlands': 19, 'Belgium': 20,
        'Switzerland': 21, 'Austria': 22, 'Portugal': 23, 'Greece': 24, 'Ireland': 25,
        'New Zealand': 26, 'Singapore': 27, 'Malaysia': 28, 'Thailand': 29,
        'Vietnam': 30, 'Philippines': 31, 'Indonesia': 32, 'Taiwan': 33,
        'Hong Kong': 34, 'Turkey': 35, 'Israel': 36, 'UAE': 37, 'Egypt': 38,
        'Morocco': 39, 'South Africa': 40, 'Nigeria': 41, 'Kenya': 42, 'Ghana': 43,
        'Argentina': 44, 'Chile': 45, 'Colombia': 46, 'Peru': 47, 'Venezuela': 48,
        'Ecuador': 49, 'Uruguay': 50, 'Paraguay': 51, 'Bolivia': 52, 'Costa Rica': 53,
        'Panama': 54, 'Jamaica': 55, 'Trinidad': 56, 'Bahamas': 57, 'Iceland': 58,
        'Finland': 59, 'Poland': 60, 'Romania': 61, 'Hungary': 62, 'Czech Republic': 63,
        'Slovakia': 64, 'Croatia': 65, 'Serbia': 66, 'Slovenia': 67, 'Bulgaria': 68,
        'Estonia': 69, 'Latvia': 70, 'Lithuania': 71, 'Ukraine': 72, 'Moldova': 73,
        'Belarus': 74, 'Kazakhstan': 75, 'Uzbekistan': 76, 'Kyrgyzstan': 77,
        'Tajikistan': 78, 'Armenia': 79, 'Georgia': 80, 'Azerbaijan': 81, 'Cyprus': 82,
        'Malta': 83, 'Luxembourg': 84, 'Monaco': 85, 'Andorra': 86, 'San Marino': 87,
        'Vatican City': 88, 'Liechtenstein': 89, 'Montenegro': 90, 'Albania': 91,
        'North Macedonia': 92, 'Kosovo': 93, 'Bosnia': 94, 'Qatar': 95, 'Kuwait': 96,
        'Bahrain': 97, 'Oman': 98, 'Jordan': 99, 'Lebanon': 100, 'Iraq': 101,
        'Yemen': 102, 'Syria': 103, 'Afghanistan': 104, 'Pakistan': 105, 'Nepal': 106,
        'Bhutan': 107, 'Sri Lanka': 108, 'Maldives': 109, 'Other': 110
    },
    'Most_Used_Platform': {
        'Instagram': 0, 'Twitter': 1, 'TikTok': 2, 'YouTube': 3, 'Facebook': 4,
        'LinkedIn': 5, 'Snapchat': 6, 'LINE': 7, 'KakaoTalk': 8, 'VKontakte': 9,
        'WhatsApp': 10, 'WeChat': 11, 'Other': 12
    },
    'Affects_Academic_Performance': {
        'Yes': 1, 'No': 0,
    },
    'Relationship_Status': {
        'Single': 0, 'In Relationship': 1, 'Complicated': 2, 'Other': 3
    },
}



def load_data(raw_data_filepath):
    """
    Carrega o dataset a partir do caminho especificado.
    """
    try:
        df = pd.read_csv(raw_data_filepath)
        logging.info(f"Dataset carregado com sucesso de: {raw_data_filepath}")
        return df
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo não foi encontrado em {raw_data_filepath}")
        return None
    except Exception as e:
        logging.error(f"Erro ao carregar o dataset: {e}")
        return None

def clean_data(df):
    """
    Realiza a limpeza básica do dataset:
    - Remove linhas duplicadas.
    - Lida com valores ausentes.
    """
    if df is None:
        return None

    logging.info("Iniciando a limpeza dos dados...")

    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_rows:
        logging.info(f"Removidas {initial_rows - df.shape[0]} linhas duplicadas.")

    # Preenche valores ausentes. Para colunas categóricas que serão mapeadas,
    # valores ausentes podem ser mapeados para um novo número ou preenchidos
    # com a categoria mais frequente antes do mapeamento.
    # Por enquanto, mantemos fillna(0) para numéricas, e o mapeamento abaixo tratará strings.
    df.fillna(0, inplace=True) # Exemplo: preenche NaNs em numéricas com 0
    if df.isnull().sum().sum() > 0:
        logging.warning("Ainda há valores ausentes após o preenchimento inicial. Verifique as colunas individualmente.")
        logging.info(f"Valores ausentes por coluna:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    else:
        logging.info("Nenhum valor ausente encontrado após o tratamento inicial.")


    logging.info("Limpeza de dados concluída.")
    return df

def preprocess_data(df):
    """
    Realiza o pré-processamento dos dados:
    - Codificação manual de variáveis categóricas para numéricas.
    """
    if df is None:
        return None

    logging.info("Iniciando o pré-processamento dos dados (Codificação Manual)...")

    for col, mapping in CATEGORICAL_MAPPINGS.items():
        if col in df.columns:
            # Use .map() para aplicar o mapeamento
            # .fillna(-1) pode ser usado para categorias não encontradas no mapeamento,
            # ou para preencher NaNs que possam existir e não foram tratados antes.
            # Se uma categoria não estiver no `mapping`, ela se tornará NaN.
            df[col] = df[col].map(mapping).fillna(-1) # Usando -1 para valores não mapeados/NaNs
            logging.info(f"Coluna '{col}' mapeada para numérico. Valores não mapeados/NaNs viraram -1.")
        else:
            logging.warning(f"Coluna '{col}' não encontrada no DataFrame para mapeamento. Pulando.")

    # Após o mapeamento, verificar se ainda há colunas de objeto (string) não tratadas
    # (que não estão no CATEGORICAL_MAPPINGS) e que não sejam a coluna alvo.
    non_numeric_cols_after_mapping = df.select_dtypes(exclude=['number']).columns.tolist()
    if 'Addicted_Score' in non_numeric_cols_after_mapping:
        non_numeric_cols_after_mapping.remove('Addicted_Score')
    
    if non_numeric_cols_after_mapping:
        logging.error(f"Atenção: As seguintes colunas ainda não são numéricas após o mapeamento e podem causar erros: {non_numeric_cols_after_mapping}")
        # Forçar conversão para numérico para qualquer coluna restante de objeto
        for col in non_numeric_cols_after_mapping:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(0, inplace=True) # Preencher NaNs criados pela coerção
            logging.warning(f"Coluna '{col}' forçada para tipo numérico e NaNs preenchidos com 0.")

    logging.info("Pré-processamento de dados concluído.")
    return df

def save_processed_data(df, processed_data_filepath):
    """
    Salva o dataset processado.
    """
    if df is None:
        logging.error("Não há dados para salvar. O DataFrame de entrada é None.")
        return

    try:
        os.makedirs(os.path.dirname(processed_data_filepath), exist_ok=True)
        df.to_csv(processed_data_filepath, index=False)
        logging.info(f"Dataset processado salvo com sucesso em: {processed_data_filepath}")
    except Exception as e:
        logging.error(f"Erro ao salvar o dataset processado: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    project_root = 'C:/Users/leonardo.romariz/Documents/mvp-qualidade-seguranca-sistemas-inteligentes/mvp-qualidade-seguranca-sistemas-inteligentes'

    raw_data_filename = 'students_social_media_addiction.csv'
    raw_data_path = os.path.join(project_root, 'data', 'raw', raw_data_filename)

    processed_data_filename = 'students_social_media_addiction_processed.csv'
    processed_data_path = os.path.join(project_root, 'data', 'processed', processed_data_filename)

    logging.info("Iniciando o pipeline de processamento de dados...")

    data = load_data(raw_data_path)

    if data is not None:
        cleaned_data = clean_data(data.copy())

        if cleaned_data is not None:
            processed_data = preprocess_data(cleaned_data.copy())

            if processed_data is not None:
                save_processed_data(processed_data, processed_data_path)
            else:
                logging.error("Pré-processamento de dados falhou.")
        else:
            logging.error("Limpeza de dados falhou.")
    else:
        logging.error("Carregamento de dados brutos falhou.")

    logging.info("Pipeline de processamento de dados concluído.")