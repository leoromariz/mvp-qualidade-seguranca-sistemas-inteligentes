# mvp-qualidade-seguranca-sistemas-inteligentes/src/tests/test_model.py

import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.train_model import train_model, save_model, load_processed_data
# Importar os mapeamentos definidos em make_dataset.py
# É crucial que make_dataset.py tenha os mesmos mapeamentos e a lógica de preprocess_data
from data.make_dataset import clean_data, preprocess_data, CATEGORICAL_MAPPINGS

# Definir as listas de features numéricas e categóricas (strings)
# Estas listas devem ser consistentes com o que make_dataset.py e app.py esperam.
NUMERIC_FEATURES_FOR_TEST = [
    'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
    'Mental_Health_Score', 'Conflicts_Over_Social_Media',
    'SMAS', 'PHQ9', 'GAD7', 'SWLS', 'Loneliness_Scale' # Adicionei estas pois são numéricas no seu dummy data
]
STRING_CATEGORICAL_FEATURES_FOR_TEST = [
    'Gender', 'Academic_Level', 'Country', 'Most_Used_Platform',
    'Affects_Academic_Performance', 'Relationship_Status'
]


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Configura os dados de teste uma única vez para todas as funções de teste.
        Cria um dataset dummy para os testes.
        """
        # Caminho base do projeto para os testes
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        cls.test_data_path = os.path.join(project_root, 'src', 'data', 'processed', 'test_students_addiction_processed.csv')
        cls.test_model_path = os.path.join(project_root, 'models', 'test_random_forest_addiction_model.joblib')

        # Criar um DataFrame de teste que simule o 'Students Social Media Addiction.csv'
        # com a coluna alvo 'Addicted_Score'
        # As colunas categóricas AQUI ainda são STRINGS, pois o preprocess_data as converterá.
        cls.test_df = pd.DataFrame({
            'Age': [20, 22, 25, 19, 21, 23, 24, 20, 26, 22],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'Relationship_Status': ['Single', 'In Relationship', 'Complicated', 'Single', 'In Relationship', 'Complicated', 'Single', 'In Relationship', 'Complicated', 'Single'],
            'Academic_Level': ['Undergraduate', 'Graduate', 'High School', 'Undergraduate', 'Graduate', 'High School', 'Undergraduate', 'Graduate', 'High School', 'Undergraduate'],
            'Country': ['Brazil', 'USA', 'Germany', 'Brazil', 'USA', 'Germany', 'Brazil', 'USA', 'Germany', 'Brazil'],
            'Avg_Daily_Usage_Hours': [3.0, 5.5, 7.0, 2.0, 4.0, 6.0, 3.5, 4.5, 8.0, 5.0],
            'Most_Used_Platform': ['Instagram', 'TikTok', 'YouTube', 'Facebook', 'Twitter', 'Instagram', 'TikTok', 'YouTube', 'Facebook', 'Twitter'],
            'Affects_Academic_Performance': ['No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
            'Sleep_Hours_Per_Night': [7.0, 6.0, 5.0, 8.0, 7.5, 6.5, 7.0, 6.0, 5.5, 7.0],
            'Mental_Health_Score': [80, 65, 50, 90, 75, 55, 85, 70, 45, 80],
            'Conflicts_Over_Social_Media': [0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
            'Addicted_Score': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0] # 0: Não Viciado, 1: Viciado
        })

        # Processar o DataFrame de teste USANDO AS MESMAS FUNÇÕES DE make_dataset.py
        cls.test_df_processed = clean_data(cls.test_df.copy())
        cls.test_df_processed = preprocess_data(cls.test_df_processed.copy())

        # Salvar o DataFrame processado para que `load_processed_data` possa carregá-lo
        os.makedirs(os.path.dirname(cls.test_data_path), exist_ok=True)
        cls.test_df_processed.to_csv(cls.test_data_path, index=False)


    def test_a_model_training(self):
        df = load_processed_data(self.test_data_path)
        self.assertIsNotNone(df, "Falha ao carregar o dataset de teste.")

        target_column = 'Addicted_Score'
        self.assertIn(target_column, df.columns, f"Coluna alvo '{target_column}' não encontrada no DataFrame de teste.")

        model, X_test, y_test, y_pred, accuracy = train_model(df, target_column=target_column)

        self.assertIsNotNone(model, "O treinamento do modelo falhou, o modelo é None.")
        self.assertGreater(accuracy, 0.5, "Acurácia do modelo muito baixa, indicando falha no treinamento ou dados de teste.")

        save_model(model, self.test_model_path)
        self.assertTrue(os.path.exists(self.test_model_path), "O modelo treinado não foi salvo.")

    def test_b_model_loading_and_prediction(self):
        if not os.path.exists(self.test_model_path):
            self.fail("O modelo não foi salvo pelo teste de treinamento. Execute o teste de treinamento primeiro.")

        loaded_model = joblib.load(self.test_model_path)
        self.assertIsNotNone(loaded_model, "Falha ao carregar o modelo salvo.")

        df = load_processed_data(self.test_data_path)
        self.assertIsNotNone(df, "Falha ao carregar o dataset de teste para previsão.")

        target_column = 'Addicted_Score'
        X_test_predict = df.drop(columns=[target_column])
        y_test_original = df[target_column]

        # Garantir que as colunas de X_test_predict correspondem às features de treinamento
        if hasattr(loaded_model, 'feature_names_in_') and loaded_model.feature_names_in_ is not None:
            # Reindexar X_test_predict para corresponder à ordem e colunas do treino
            missing_cols = set(loaded_model.feature_names_in_) - set(X_test_predict.columns)
            for c in missing_cols:
                X_test_predict[c] = 0 # Preenche colunas que podem ter faltado no teste dummy (ex: categoria não representada)
            X_test_predict = X_test_predict[loaded_model.feature_names_in_]
        else:
            # Esta parte é um fallback crítico. A ordem das colunas DEVE ser a mesma do treinamento.
            # Baseado nas features que restaram no seu CATEGORICAL_MAPPINGS + outras numéricas
            # Esta lista precisa ser a lista EXATA de features que o modelo espera, na ordem certa.
            expected_features_fallback = [
                'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media',
                'SMAS', 'PHQ9', 'GAD7', 'SWLS', 'Loneliness_Scale', # As numéricas que estavam no seu dummy
                'Gender', 'Academic_Level', 'Country', 'Most_Used_Platform',
                'Affects_Academic_Performance', 'Relationship_Status'
            ]
            
            # Garante que X_test_predict tem todas as colunas esperadas, preenchendo com 0 se ausentes
            for col in expected_features_fallback:
                if col not in X_test_predict.columns:
                    X_test_predict[col] = 0
            # Reordena as colunas
            X_test_predict = X_test_predict[expected_features_fallback]


        predictions = loaded_model.predict(X_test_predict)
        self.assertEqual(len(predictions), len(X_test_predict), "O número de previsões não corresponde ao número de amostras.")

        accuracy = accuracy_score(y_test_original, predictions)
        self.assertGreaterEqual(accuracy, 0.5, "Acurácia da previsão com modelo carregado muito baixa.")

    @classmethod
    def tearDownClass(cls):
        # Limpeza de arquivos e diretórios criados durante os testes
        if os.path.exists(cls.test_data_path):
            os.remove(cls.test_data_path)
        if os.path.exists(cls.test_model_path):
            os.remove(cls.test_model_path)

        processed_dir = os.path.dirname(cls.test_data_path)
        # Verifica se o diretório processed está vazio antes de tentar removê-lo
        if os.path.exists(processed_dir) and not os.listdir(processed_dir):
            os.rmdir(processed_dir)
        models_dir = os.path.dirname(cls.test_model_path)
        # Verifica se o diretório models está vazio antes de tentar removê-lo
        if os.path.exists(models_dir) and not os.listdir(models_dir):
            os.rmdir(models_dir)

if __name__ == '__main__':
    unittest.main()