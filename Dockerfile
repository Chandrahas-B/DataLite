FROM python:3.7

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# Install dependencies
RUN pip install -r requirements.txt
RUN apt-get update
RUN echo "Make sure to have CUDA>11.0 for GPU support"
RUN pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

ENV PORT 8501

CMD ["streamlit","run","app.py"]
