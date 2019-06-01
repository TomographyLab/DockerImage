# Start with NVidia CUDA base image
FROM nvidia/cuda:10.0-devel-ubuntu18.04

MAINTAINER Michele Scipioni <scipioni.michele@gmail.com>

ARG PYTHON_VERSION=3.7
ARG CONDA_PYTHON_VERSION=3
ARG SRC_DIR=src/
ARG SCRIPTS_DIR=scripts/
ARG USERNAME=occiput
ARG USERID=1000
ARG GID="100"

USER root

############# Install all OS dependencies for notebook server that starts but lacks all
# features (e.g., download as all possible file formats)
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -yq dist-upgrade \
    && apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        libncurses5-dev \
        libgl1-mesa-dev \
        libx11-dev \
        libxrandr-dev \
        libxi-dev \
        libglew-dev \
        locales \
        lsof \
        nano \
        software-properties-common \
        sudo \
        wget \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

############# Configure environment
ENV CONDA_DIR=/opt/conda \
    NB_DIR=/home/$USERNAME/notebooks \
    CODE_DIR=/home/$USERNAME/code \
    SHELL=/bin/bash \
    NB_UID=$USERID \
    NB_GID=$GID \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    HOME=/home/$USERNAME \
    CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=$CUDA_HOME
ENV PATH=$PATH:$CONDA_DIR/bin:$CUDA_HOME/bin:$HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
ENV C_INCLUDE_PATH=${CUDA_HOME}/include:${C_INCLUDE_PATH}

RUN mkdir -p $CONDA_DIR && \
    mkdir -p $NB_DIR && \
    mkdir -p $CODE_DIR


############# Create USERNAME
# with name occiput user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.

# Add a script that we will use to correct permissions after running certain commands
COPY ${SCRIPTS_DIR}/fix-permissions /usr/local/bin/fix-permissions
#COPY ${SRC_DIR}/cmake-3.14.3.tar.gz ${HOME}/cmake-3.14.3.tar.gz
COPY ${SRC_DIR}/freeglut-3.0.0.tar.gz ${HOME}/freeglut-3.0.0.tar.gz
COPY ${SRC_DIR}/NiftyRec.tar.gz ${HOME}/NiftyRec.tar.gz

# Enable prompt color in the skeleton .bashrc before creating the default USERNAME
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc  \
    && echo "auth requisite pam_deny.so" >> /etc/pam.d/su  \
    && sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers  \
    && sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers  \
    && useradd -m -s /bin/bash -N -u $NB_UID $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && chown $USERNAME:$NB_GID $CONDA_DIR  \
    && chown $USERNAME:$NB_GID $NB_DIR  \
    && chown $USERNAME:$NB_GID $CODE_DIR  \
    && chmod g+w /etc/passwd  \
    && chown $USERNAME:$NB_GID /usr/local/bin/fix-permissions \
    #&& tar xzf ${HOME}/cmake-3.14.3.tar.gz -C ${HOME}  \
    #&& cd ${HOME}/cmake-3.14.3 \
    #&& ./bootstrap \
    #&& make -j8 \
    #&& make install  \
    && cd $HOME && rm ${HOME}/cmake* -rf  \
    && cd /tmp && wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh  \
    && /bin/bash Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR  \
    && rm Miniconda$CONDA_PYTHON_VERSION-latest-Linux-x86_64.sh  \
    && conda config --add channels conda-forge \
    && conda config --system --prepend channels conda-forge  \
    && conda config --system --set auto_update_conda false  \
    && conda config --system --set show_channel_urls true  \
    && conda update -yn base conda \
    && conda update -yn base --all \
    && conda install -c anaconda libboost  \
    && conda install -c conda-forge ipywidgets \
                                    ipyvolume \
                                    nodejs \
    && conda install --quiet --yes \
                      cmake=3.14.4 \
                      h5py \
                      jupyterlab \
                      matplotlib \
                      notebook \
                      scikit-learn \
    && conda update --all --quiet --yes  \
    && conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned  \
    && conda clean --all -f -y \
    && rm -rf /var/cache/apk/*   \
    && find $CONDA_DIR \
      \( -type d -a -name test -o -name tests \) \
      -o \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
      -exec rm -rf '{}' + \
    && npm cache clean --force  \
    && jupyter notebook --generate-config  \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    && jupyter labextension install ipyvolume \
    && rm -rf $CONDA_DIR/share/jupyter/lab/staging  \
    && rm -rf ${HOME}/.cache/yarn  \
    && fix-permissions $CONDA_DIR  \
    && fix-permissions ${HOME} \
    && pip install --upgrade pip  \
    && pip install --no-cache-dir \
                      dcmstack \
                      ipy_table \
                      nibabel \
                      nipy \
                      pycuda \
                      pydicom \
                      scikit-cuda \
                      scikit-image \
                      svgwrite \
    && rm -rf ~/.cache/pip \
    && fix-permissions ${HOME}  \
    && tar xzf ${HOME}/freeglut-3.0.0.tar.gz -C ${HOME} \
    && cd ${HOME}/freeglut-3.0.0 \
    && cmake . \
    && make all -j8 \
    && make install \
    && cd $HOME && rm ${HOME}/freeglut* -rf \
    && tar xzf ${HOME}/NiftyRec.tar.gz -C /opt \
    && cd $HOME && rm ${HOME}/NiftyRec* -rf \
    && fix-permissions $HOME  \
    && fix-permissions "$(dirname $CONDA_DIR)"  \
    && fix-permissions "$(dirname $NB_DIR)" \
    && fix-permissions "$(dirname $CODE_DIR)"

############

EXPOSE 8888
EXPOSE 8080
WORKDIR $HOME

############# ENV variables setup
ENV BROWSER=google-chrome
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/NiftyRec/lib
ENV PYTHONPATH=$PYTHONPATH:${HOME}/code/tomolab

############# Configure container startup
CMD ["start-notebook.sh"]

# Add local files as late as possible to avoid cache busting
COPY --chown=occiput:100 $SCRIPTS_DIR/start.sh /usr/local/bin/
COPY --chown=occiput:100 $SCRIPTS_DIR/start-notebook.sh /usr/local/bin/
COPY --chown=occiput:100 $SCRIPTS_DIR/start-singleuser.sh /usr/local/bin/
COPY --chown=occiput:100 config/jupyter_notebook_config.py /etc/jupyter/
RUN fix-permissions /etc/jupyter/

# Switch back to occiput user to avoid accidental container runs as root
USER $USERID
