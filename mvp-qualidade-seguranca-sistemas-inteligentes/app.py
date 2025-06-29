# mvp-qualidade-seguranca-sistemas-inteligentes/src/app.py

from flask import Flask, request, jsonify, redirect, url_for
from flask.views import MethodView
from flask_smorest import Api, Blueprint, abort
import joblib
import pandas as pd
import logging
import os
from marshmallow import Schema, fields, validate

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuração do Flask-Smorest para OpenAPI/Swagger ---
app.config["API_TITLE"] = "API de Previsão de Vício em Mídias Sociais"
app.config["API_VERSION"] = "1.0"
app.config["OPENAPI_VERSION"] = "3.0.2"
app.config["OPENAPI_URL_PREFIX"] = "/" # Onde o JSON do OpenAPI será servido
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui" # Onde a interface do Swagger UI será servida
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/" # CDN para o Swagger UI

api = Api(app)

# --- Carregamento do Modelo ---
MODEL_PATH = 'C:/Users/leonardo.romariz/Documents/mvp-qualidade-seguranca-sistemas-inteligentes/mvp-qualidade-seguranca-sistemas-inteligentes/models/random_forest_addiction_model.joblib'
model = None

# IMPORTANTE: ESTES MAPEAMENTOS DEVEM SER EXATAMENTE OS MESMOS DE make_dataset.py
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
    'Avg_Daily_Usage_Hours': {},
    'Most_Used_Platform': {
        'Instagram': 0, 'Twitter': 1, 'TikTok': 2, 'YouTube': 3, 'Facebook': 4,
        'LinkedIn': 5, 'Snapchat': 6, 'LINE': 7, 'KakaoTalk': 8, 'VKontakte': 9,
        'WhatsApp': 10, 'WeChat': 11, 'Other': 12
    },
    'Affects_Academic_Performance': {
        'Yes': 1, 'No': 0,
    },
    'Sleep_Hours_Per_Night': {},
    'Mental_Health_Score': {},
    'Relationship_Status': {
        'Single': 0, 'In Relationship': 1, 'Complicated': 2, 'Other': 3
    },
    'Conflicts_Over_Social_Media': {}
}

# Separar features numéricas e categóricas com base na sua estrutura de mapeamento
NUMERIC_FEATURES = [
    'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
    'Mental_Health_Score', 'Conflicts_Over_Social_Media'
]
STRING_CATEGORICAL_FEATURES = [
    'Gender', 'Academic_Level', 'Country', 'Most_Used_Platform',
    'Affects_Academic_Performance', 'Relationship_Status'
]

def load_model():
    """Carrega o modelo treinado."""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logging.info(f"Modelo carregado com sucesso de: {MODEL_PATH}")
    except FileNotFoundError as e:
        logging.error(f"Erro: {e}")
        model = None
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo: {e}", exc_info=True)
        model = None

with app.app_context():
    load_model()

# --- Definição dos Schemas Marshmallow para entrada e saída ---

class PredictionInputSchema(Schema):
    # Campos Numéricos
    Age = fields.Integer(required=True)
    Avg_Daily_Usage_Hours = fields.Float(required=True)
    Sleep_Hours_Per_Night = fields.Float(required=True)
    Mental_Health_Score = fields.Integer(required=True)
    Conflicts_Over_Social_Media = fields.Float(required=True)

    # Campos Categóricos (Strings com validação de lista de opções)
    Gender = fields.String(required=True, validate=validate.OneOf(list(CATEGORICAL_MAPPINGS['Gender'].keys())))
    Academic_Level = fields.String(required=True, validate=validate.OneOf(list(CATEGORICAL_MAPPINGS['Academic_Level'].keys())))
    Country = fields.String(required=True, validate=validate.OneOf(list(CATEGORICAL_MAPPINGS['Country'].keys())))
    Most_Used_Platform = fields.String(required=True, validate=validate.OneOf(list(CATEGORICAL_MAPPINGS['Most_Used_Platform'].keys())))
    Affects_Academic_Performance = fields.String(required=True, validate=validate.OneOf(list(CATEGORICAL_MAPPINGS['Affects_Academic_Performance'].keys())))
    Relationship_Status = fields.String(required=True, validate=validate.OneOf(list(CATEGORICAL_MAPPINGS['Relationship_Status'].keys())))

    # Occupation e Affiliation foram removidos daqui


class PredictionOutputSchema(Schema):
    prediction = fields.String()
    predicted_class = fields.Integer()
    confidence = fields.Float()

# --- Blueprint para as rotas da API ---
blp = Blueprint(
    "Prediction", "prediction", url_prefix="/prediction",
    description="Operações para prever o nível de vício em mídias sociais"
)

@blp.route("/")
class PredictionResource(MethodView):

    @blp.arguments(PredictionInputSchema)
    @blp.response(200, PredictionOutputSchema)
    def post(self, new_data):
        """
        Realiza a previsão do nível de vício em mídias sociais.
        """
        if model is None:
            logging.error("Tentativa de previsão com modelo não carregado.")
            abort(500, message='Modelo de previsão não carregado. Verifique os logs do servidor.')

        try:
            logging.info(f"Dados recebidos para previsão: {new_data}")

            processed_data = {}

            # Processar colunas categóricas (mapeamento de string para número)
            for col in STRING_CATEGORICAL_FEATURES:
                if col in new_data and new_data[col] in CATEGORICAL_MAPPINGS[col]:
                    processed_data[col] = CATEGORICAL_MAPPINGS[col][new_data[col]]
                else:
                    processed_data[col] = -1
                    if col not in new_data:
                        logging.warning(f"Coluna categórica '{col}' esperada, mas não encontrada na entrada. Adicionada com valor padrão -1.")
                    else:
                        logging.warning(f"Valor '{new_data[col]}' para '{col}' não encontrado no mapeamento. Usando valor padrão -1.")

            # Processar colunas numéricas (passar diretamente)
            for col in NUMERIC_FEATURES:
                if col in new_data:
                    try:
                        processed_data[col] = float(new_data[col])
                    except ValueError:
                        logging.error(f"Valor inválido para coluna numérica '{col}': {new_data[col]}. Usando 0.")
                        processed_data[col] = 0.0
                else:
                    processed_data[col] = 0.0
                    logging.warning(f"Coluna numérica '{col}' esperada, mas não encontrada na entrada. Adicionada com valor padrão 0.0.")

            # Criar DataFrame final para previsão
            if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                expected_features_order = model.feature_names_in_.tolist()
            else:
                # Fallback - Garanta que esta lista corresponde EXATAMENTE às features de treinamento do seu modelo, na ORDEM CORRETA.
                expected_features_order = [
                    'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media',
                    'Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 'Affects_Academic_Performance', 'Relationship_Status'
                    # Occupation e Affiliation foram removidos daqui também
                ]
                logging.warning("model.feature_names_in_ não disponível. Usando uma ordem predefinida de features. ISSO PODE CAUSAR PROBLEMAS DE ORDENAÇÃO E RESULTADOS INCORRETOS.")

            # Garante que todas as features esperadas pelo modelo estão no processed_data
            # e que o DataFrame final para previsão tem a ordem correta.
            final_input_data = {}
            for feature in expected_features_order:
                if feature in processed_data:
                    final_input_data[feature] = processed_data[feature]
                else:
                    # Caso uma feature esperada pelo modelo não tenha sido processada
                    # (ex: não estava no schema nem nas listas NUMERIC_FEATURES/STRING_CATEGORICAL_FEATURES)
                    # Isso pode indicar um erro de alinhamento entre o schema/listas e o modelo.
                    final_input_data[feature] = 0 # Valor padrão (pode ser -1 dependendo do modelo)
                    logging.warning(f"Feature '{feature}' esperada pelo modelo, mas não encontrada no processed_data. Adicionada com valor padrão 0.")


            input_for_prediction = pd.DataFrame([final_input_data])[expected_features_order]

            prediction = model.predict(input_for_prediction)
            prediction_proba = model.predict_proba(input_for_prediction)

            predicted_class = int(prediction[0])
            confidence = float(prediction_proba[0][predicted_class])

            class_mapping = {0: 'Não Viciado', 1: 'Viciado'}
            predicted_label = class_mapping.get(predicted_class, f'Classe Desconhecida: {predicted_class}')

            logging.info(f"Previsão realizada: Classe {predicted_class} ({predicted_label}) com confiança {confidence:.4f}")

            return {
                'prediction': predicted_label,
                'predicted_class': predicted_class,
                'confidence': confidence
            }

        except Exception as e:
            logging.error(f"Erro inesperado durante a previsão: {e}", exc_info=True)
            abort(500, message=f'Erro interno do servidor: {e}')

# --- Registrar o Blueprint na API principal ---
api.register_blueprint(blp)

# --- Redirecionamento da rota raiz para o Swagger UI ---
@app.route('/')
def redirect_to_swagger():
    """
    Redireciona a rota raiz para a interface do Swagger UI.
    """
    return redirect(url_for('api-docs.openapi_swagger_ui'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)