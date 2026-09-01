# CM3070 Telegram financial-advisor bot.
#
# Serves the screened recommendation over the Telegram Bot API. Deliberately
# slim: the integration layer's forecaster-bridge import is deferred behind
# TYPE_CHECKING, so the screen runs without torch or the forecaster package.
#
# The cluster has no PyPI egress and no market-data egress, so dependencies and
# the M5 price cache are both baked in at build time.
#
# Build from the repository root:
#   docker build -f src/feature-prototype/Dockerfile -t <registry>/forex-cm3070-advisor:latest .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY src/feature-prototype/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# The advisor imports dqnpf.{action_mapper,config,integration}; the rest of the
# package comes along so the module graph resolves, but nothing heavy is loaded.
COPY src/dqnpf/ /app/dqnpf/
COPY src/feature-prototype/ /app/feature-prototype/

# Fail the build rather than the pod if the import graph regresses.
RUN python -c "\
import sys; sys.path.insert(0, '/app/feature-prototype'); \
from dqnpf.integration import IntegrationLayer; \
import telegram_advisor_bot; \
assert 'torch' not in sys.modules, 'torch leaked into the advisor image'; \
print('import smoke OK')"

CMD ["python", "feature-prototype/telegram_advisor_bot.py"]
