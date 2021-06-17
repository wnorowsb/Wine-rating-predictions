FROM python:3.7-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# copy and insall requirements only
COPY . /opt/pipeline/
RUN pip install -r /opt/pipeline/requirements.txt

WORKDIR /opt/pipeline/