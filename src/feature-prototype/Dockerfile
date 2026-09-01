FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY src/feature-prototype/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY src/dqnpf/ /app/dqnpf/
COPY src/feature-prototype/ /app/feature-prototype/

RUN python -c "\
import sys; sys.path.insert(0, '/app/feature-prototype'); \
from dqnpf.integration import IntegrationLayer; \
import telegram_advisor_bot; \
assert 'torch' not in sys.modules, 'torch leaked into the advisor image'; \
print('import smoke OK')"

CMD ["python", "feature-prototype/telegram_advisor_bot.py"]
