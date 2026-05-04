# syntax=docker/dockerfile:1

# Build on Apple Silicon with:
#   docker build --platform linux/amd64 -t rtcog .
#
# Run a quick smoke test with:
#   docker run --rm --platform linux/amd64 rtcog
FROM afni/afni_make_build:AFNI_25.2.03

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV CONDA_DIR=/opt/conda \
    CONDA_ENV=rtcog_min \
    PYTHONUNBUFFERED=1
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Install packages needed for miniconda
RUN apt-get update && \
    apt-get install -y --no-install-recommends bzip2 ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

WORKDIR /app

# Install conda environment
COPY minimal_env.yaml .
RUN conda env create -f minimal_env.yaml && \
    conda clean -afy

ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV}/bin:${CONDA_DIR}/bin:${PATH}

# Copy only the package files needed at runtime
COPY pyproject.toml README.md ./
COPY rtcog ./rtcog

RUN pip install --no-cache-dir -e . && \
    mkdir -p /app/output && \
    chown -R afni_user /app

USER afni_user

ENTRYPOINT ["rtcog_min"]
CMD ["--help"]
