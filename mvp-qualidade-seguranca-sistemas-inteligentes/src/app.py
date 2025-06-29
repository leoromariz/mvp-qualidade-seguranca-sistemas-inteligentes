# mvp-qualidade-seguranca-sistemas-inteligentes/src/app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import os

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = 'C:/Users/leonardo.romariz/Documents/mvp-qualidade-seguranca-sistemas-inteligentes/mvp-qualidade-seguranca-sistemas-inteligentes/models/random_forest_addiction_model.joblib'
model = None

# IMPORTANTE: ESTES MAPEAMENTOS DEVEM SER EXATAMENTE OS MESMOS DE make_dataset.py
CATEGORICAL_MAPPINGS = {
    'Gender': {
        'Male': 0,
        'Female': 1,
        'Other': 2
    },
    'Relationship_Status': {
        'Single': 0,
        'In Relationship': 1,
        'Complicated': 2,
        'Other': 3
        # Adicione outros se existirem, ex: 'Divorced': 3
    },
    'Most_Used_Platform': {
    'Instagram': 0,
    'Twitter': 1,
    'TikTok': 2,
    'YouTube': 3,
    'Facebook': 4,
    'LinkedIn': 5,
    'Snapchat': 6,
    'LINE': 7,
    'KakaoTalk': 8,
    'VKontakte': 9,
    'WhatsApp': 10,
    'WeChat': 11,
    'Other': 12
    },
    'Academic_Level': {
        'Undergraduate': 0,
        'Graduate': 1,
        'High School': 2,
        'Other': 3
    },
    'Affects_Academic_Performance': {
        'Yes': 1,
        'No': 0,
    },
    'Country': {
    'Bangladesh': 0,
    'India': 1,
    'USA': 2,
    'UK': 3,
    'Canada': 4,
    'Australia': 5,
    'Germany': 6,
    'Brazil': 7,
    'Japan': 8,
    'South Korea': 9,
    'France': 10,
    'Spain': 11,
    'Italy': 12,
    'Mexico': 13,
    'Russia': 14,
    'China': 15,
    'Sweden': 16,
    'Norway': 17,
    'Denmark': 18,
    'Netherlands': 19,
    'Belgium': 20,
    'Switzerland': 21,
    'Austria': 22,
    'Portugal': 23,
    'Greece': 24,
    'Ireland': 25,
    'New Zealand': 26,
    'Singapore': 27,
    'Malaysia': 28,
    'Thailand': 29,
    'Vietnam': 30,
    'Philippines': 31,
    'Indonesia': 32,
    'Taiwan': 33,
    'Hong Kong': 34,
    'Turkey': 35,
    'Israel': 36,
    'UAE': 37,
    'Egypt': 38,
    'Morocco': 39,
    'South Africa': 40,
    'Nigeria': 41,
    'Kenya': 42,
    'Ghana': 43,
    'Argentina': 44,
    'Chile': 45,
    'Colombia': 46,
    'Peru': 47,
    'Venezuela': 48,
    'Ecuador': 49,
    'Uruguay': 50,
    'Paraguay': 51,
    'Bolivia': 52,
    'Costa Rica': 53,
    'Panama': 54,
    'Jamaica': 55,
    'Trinidad': 56,
    'Bahamas': 57,
    'Iceland': 58,
    'Finland': 59,
    'Poland': 60,
    'Romania': 61,
    'Hungary': 62,
    'Czech Republic': 63,
    'Slovakia': 64,
    'Croatia': 65,
    'Serbia': 66,
    'Slovenia': 67,
    'Bulgaria': 68,
    'Estonia': 69,
    'Latvia': 70,
    'Lithuania': 71,
    'Ukraine': 72,
    'Moldova': 73,
    'Belarus': 74,
    'Kazakhstan': 75,
    'Uzbekistan': 76,
    'Kyrgyzstan': 77,
    'Tajikistan': 78,
    'Armenia': 79,
    'Georgia': 80,
    'Azerbaijan': 81,
    'Cyprus': 82,
    'Malta': 83,
    'Luxembourg': 84,
    'Monaco': 85,
    'Andorra': 86,
    'San Marino': 87,
    'Vatican City': 88,
    'Liechtenstein': 89,
    'Montenegro': 90,
    'Albania': 91,
    'North Macedonia': 92,
    'Kosovo': 93,
    'Bosnia': 94,
    'Qatar': 95,
    'Kuwait': 96,
    'Bahrain': 97,
    'Oman': 98,
    'Jordan': 99,
    'Lebanon': 100,
    'Iraq': 101,
    'Yemen': 102,
    'Syria': 103,
    'Afghanistan': 104,
    'Pakistan': 105,
    'Nepal': 106,
    'Bhutan': 107,
    'Sri Lanka': 108,
    'Maldives': 109,
    'Other': 110
    }
}


def load_model():
    """Carrega o modelo treinado."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logging.info(f"Modelo carregado com sucesso de: {MODEL_PATH}")
    except FileNotFoundError:
        logging.error(f"Erro: O arquivo do modelo não foi encontrado em {MODEL_PATH}")
        model = None
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo: {e}")
        model = None

with app.app_context():
    load_model()

@app.route('/')
def home():
    return "Bem-vindo à API de Previsão de Nível de Vício em Mídias Sociais!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar previsões.
    Espera um JSON com os dados do aluno (pode ter strings para categóricas).
    """
    if model is None:
        logging.error("Tentativa de previsão com modelo não carregado.")
        return jsonify({'error': 'Modelo de previsão não carregado. Verifique os logs do servidor.'}), 500

    try:
        data = request.get_json(force=True)
        logging.info(f"Dados recebidos para previsão: {data}")

        input_df = pd.DataFrame([data]) # Cria um DataFrame a partir do dicionário de entrada

        # Aplicar os mesmos mapeamentos categóricos que foram usados em make_dataset.py
        for col, mapping in CATEGORICAL_MAPPINGS.items():
            if col in input_df.columns:
                # Mapeia, tratando valores não encontrados ou NaNs
                input_df[col] = input_df[col].map(mapping).fillna(-1) # -1 para valores não mapeados/NaNs
            else:
                # Se a coluna categórica esperada não está no input, adiciona com valor padrão (-1)
                input_df[col] = -1
                logging.warning(f"Coluna '{col}' esperada para mapeamento, mas não encontrada na entrada. Adicionada com valor padrão -1.")

        # Verificar se, após o mapeamento, ainda restam colunas não numéricas
        non_numeric_cols = input_df.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            logging.error(f"Atenção: As seguintes colunas ainda não são numéricas na entrada da API: {non_numeric_cols}")
            # Forçar conversão para numérico para qualquer coluna restante de objeto
            for col in non_numeric_cols:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                input_df[col].fillna(0, inplace=True)
                logging.warning(f"Coluna '{col}' forçada para tipo numérico e NaNs preenchidos com 0 na API.")

        # Obter a lista de features que o modelo foi treinado para esperar
        # É CRÍTICO que as colunas e a ordem sejam as mesmas usadas durante o treinamento.
        # model.feature_names_in_ é a forma mais segura de obter isso do modelo salvo.
        if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
            expected_features_order = model.feature_names_in_
        else:
            # Fallback se feature_names_in_ não estiver disponível (menos robusto).
            # Liste todas as colunas NUMÉRICAS originais MAIS as que serão criadas pelo mapeamento,
            # e EXCLUA a coluna alvo 'Addicted_Score'.
            # Esta lista DEVE estar na ordem que o modelo foi treinado.
            # Você pode precisar executar make_dataset.py -> train_model.py e imprimir X_train.columns para verificar a ordem.
            expected_features_order = [
                'Age', 'SMAS', 'PHQ9', 'GAD7', 'SWLS', 'Loneliness_Scale',
                'Gender', 'Relationship Status', 'Occupation', 'Affiliation'
            ]
            logging.warning("model.feature_names_in_ não disponível. Usando uma ordem predefinida de features. Isso pode causar problemas de ordenação.")


        # Garanta que o DataFrame de entrada tenha todas as colunas esperadas com 0s ou -1s se ausentes
        for col in expected_features_order:
            if col not in input_df.columns:
                input_df[col] = 0 # ou -1, dependendo do que você quer para features ausentes
                logging.warning(f"Coluna '{col}' esperada mas não encontrada na entrada JSON. Adicionada com valor padrão 0.")

        # Selecionar e reordenar as colunas para corresponder ao treinamento
        input_for_prediction = input_df[expected_features_order]


        prediction = model.predict(input_for_prediction)
        prediction_proba = model.predict_proba(input_for_prediction)

        predicted_class = int(prediction[0])
        confidence = float(prediction_proba[0][predicted_class])

        # Exemplo de mapeamento de classes para 'Addicted_Score'.
        # AJUSTE ISTO CONFORME AS CLASSES REAIS DO SEU 'Addicted_Score' (ex: 0=Não, 1=Sim)
        class_mapping = {0: 'Não Viciado', 1: 'Viciado'} # Adapte para as classes do seu dataset
        predicted_label = class_mapping.get(predicted_class, f'Classe Desconhecida: {predicted_class}')


        logging.info(f"Previsão realizada: Classe {predicted_class} ({predicted_label}) com confiança {confidence:.4f}")

        return jsonify({
            'prediction': predicted_label,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except KeyError as e:
        logging.error(f"Erro de chave ausente no JSON de entrada: {e}")
        return jsonify({'error': f'Dados de entrada inválidos. Chave ausente: {e}. Verifique se todas as colunas esperadas estão presentes e corretas.'}), 400
    except Exception as e:
        logging.error(f"Erro inesperado durante a previsão: {e}", exc_info=True)
        return jsonify({'error': f'Erro interno do servidor: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)