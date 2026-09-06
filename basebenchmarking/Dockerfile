FROM continuumio/miniconda3:latest

WORKDIR /app

# Create a conda environment with Python, AROSICS (which installs GDAL correctly on Linux)
# We also install OpenCV, PyTorch, Octave and oct2py via conda-forge
RUN conda create -n benchmark_env -c conda-forge \
    python=3.10 \
    arosics \
    gdal \
    rasterio \
    pyproj \
    opencv \
    "pytorch>=2.0" \
    torchvision \
    kornia \
    flask \
    octave \
    oct2py \
    scipy \
    requests \
    && conda clean -a -y

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "benchmark_env"]
CMD ["python", "-m", "pipeline.runner"]

