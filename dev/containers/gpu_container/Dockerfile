# make nvidia container
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04
SHELL ["/bin/bash","-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install git build-essential wget zip libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl -y
WORKDIR /root
USER root
RUN mkdir /pyenv
RUN chmod 2777 /pyenv
ENV PYENV_ROOT /pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:UTF-8
ENV LC_ALL en_US.UTF-8
RUN git clone https://github.com/pyenv/pyenv.git /pyenv
RUN pyenv install 3.10.5
RUN pyenv global 3.10.5
RUN pyenv rehash
# download stable diffusion
RUN git clone https://github.com/CompVis/stable-diffusion.git
WORKDIR /root/stable-diffusion
RUN pip install -e .
# download models
RUN wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt -O /root/stable-diffusion/sd-v1-4.ckpt
RUN ls -la
RUN mkdir -p /root/stable-diffusion/models/ldm/stable-diffusion-v1
RUN ln -s /root/stable-diffusion/sd-v1-4.ckpt /root/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt

# install our package
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry config installer.parallel true
RUN mkdir /root/strd
COPY sources/pyproject.toml /root/strd/pyproject.toml
COPY sources/poetry.lock /root/strd/poetry.lock
WORKDIR /root/strd
RUN poetry cache clear pypi --all
RUN poetry lock
RUN poetry install --no-root
COPY sources /root/strd
RUN poetry install
ENV PYTHONPATH=/root/stable-diffusion

#RUN nvidia-smi
#RUN python -m pinject_design.run_config_utils run_injected get strd.adain_tests.converted strd.designs.default_design
