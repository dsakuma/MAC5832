FROM jupyter/datascience-notebook:latest
MAINTAINER Daniel Sakuma "dsakuma@gmail.com"

USER jovyan

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN git clone --recursive https://github.com/dmlc/xgboost && \
    cd xgboost && \
    make -j4 && \
    cd python-package; python setup.py install

CMD ["start-notebook.sh", "--NotebookApp.token=''"]
