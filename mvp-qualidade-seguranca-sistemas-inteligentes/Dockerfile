# mvp-qualidade-seguranca-sistemas-inteligentes/Dockerfile

# Use uma imagem base Python oficial
FROM python:3.9-slim-buster

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copia o arquivo de requisitos e instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação
COPY . .

# Cria os diretórios necessários (data/raw, data/processed, models)
# Garante que eles existam antes que os scripts tentem acessá-los
RUN mkdir -p data/raw data/processed models

# Copia o dataset para o local esperado
# O nome do arquivo foi ajustado para o seu dataset
COPY Students\ Social\ Media\ Addiction.csv data/raw/Students\ Social\ Media\ Addiction.csv

# Expõe a porta em que a aplicação Flask será executada
EXPOSE 5000

# Comando para rodar a aplicação usando Gunicorn
# Gunicorn é um servidor WSGI para Python, mais robusto para produção do que o servidor de desenvolvimento do Flask.
# O '--bind 0.0.0.0:5000' faz com que o servidor escute em todas as interfaces de rede na porta 5000.
# O 'app:app' especifica que o aplicativo Flask é chamado 'app' dentro do módulo 'app.py'.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.app:app"]