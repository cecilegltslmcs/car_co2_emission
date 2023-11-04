FROM python:3.11.6-slim

LABEL maintainer="Cécile Guillot <cecile@guillot-salmacis.com>"

EXPOSE 9696

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "predict.py", "random_forest.bin", "./"]

RUN pip install pipenv && \
    pipenv install --system --deploy

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
