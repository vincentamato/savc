from setuptools import setup, find_packages

setup(
    name="savc",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "pretty_midi",
        "pillow",
        "tqdm",
        "wandb",
        "numpy",
        "accelerate"
    ],
    author="Vincent",
    description="Style-Aware Visual Composer",
    python_requires=">=3.8",
)