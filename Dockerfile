FROM gauss-avs

COPY requirements.txt /app/

RUN conda run -n gaussian_splatting pip install --upgrade pip
RUN conda run -n gaussian_splatting pip install -r /app/requirements.txt
RUN conda run -n gaussian_splatting pip install ipdb torch_ema lpips
RUN conda run -n gaussian_splatting pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

CMD ["/bin/bash"]
