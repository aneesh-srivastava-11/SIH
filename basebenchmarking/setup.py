from setuptools import setup, find_packages

setup(
    name="image-registration-benchmark",
    version="1.0.0",
    description="Off-the-shelf Image Registration Benchmarking Platform",
    author="SIH Development Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "pandas>=1.4.0",
        "flask>=2.2.0",
        "pyyaml>=6.0",
        "Pillow>=9.0.0",
    ],
    entry_points={
        "console_scripts": [
            "run-benchmark=pipeline.runner:main",
            "run-dashboard=dashboard.app:main",
        ],
    },
)
