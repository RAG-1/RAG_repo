FROM python:3.12.7 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

FROM python:3.12.7-slim
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .

# 문서용이지만 실제 리스닝 포트와 맞춰두면 헷갈림이 줄어듭니다
EXPOSE 8080

# 올바른 명령어로 수정
CMD [".venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
