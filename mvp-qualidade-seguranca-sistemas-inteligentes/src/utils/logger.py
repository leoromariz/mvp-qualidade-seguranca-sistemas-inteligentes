# mvp-qualidade-seguranca-sistemas-inteligentes/src/utils/logger.py

import logging

def setup_logging(log_file='app.log', level=logging.INFO):
    """
    Configura o sistema de logging para a aplicação.
    As mensagens de log serão enviadas para o console e para um arquivo.
    """
    # Remove qualquer handler existente para evitar duplicação de logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging configurado. Logs serão salvos em '{log_file}' com nível '{logging.getLevelName(level)}'.")

# Exemplo de uso:
if __name__ == "__main__":
    setup_logging()
    logging.info("Esta é uma mensagem de informação.")
    logging.warning("Esta é uma mensagem de aviso.")
    logging.error("Esta é uma mensagem de erro.")
    try:
        1 / 0
    except ZeroDivisionError:
        logging.exception("Ocorreu um erro de divisão por zero.")