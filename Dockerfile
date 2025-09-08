ARG PYTHON_VERSION=3.12-slim
FROM python:${PYTHON_VERSION}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

COPY requirements.txt /work/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /work
RUN pip install -e .

# default: run-all with example path (user should mount data/out)
CMD ["egr-cmb-run-all", "--data", "data/COM_PowerSpect_CMB-TT-full_R3.01.txt", "--outdir", "out_cmb"]
