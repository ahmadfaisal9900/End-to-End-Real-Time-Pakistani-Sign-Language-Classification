import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "Sign-Language-MLOps"
AUTHOR_USER_NAME = "yourusername"  # Replace with your username
SRC_REPO = "Sign_Language_Classification"
AUTHOR_EMAIL = "your.email@example.com"  # Replace with your email

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A machine learning project for sign language classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "datasets>=3.0.0",
        "evaluate>=0.4.0",
        "opencv-python",
        "numpy",
        "tqdm",
        "pyyaml",
        "matplotlib",
        "scikit-learn",
        "pillow",
        "pytest",
        "dvc"
    ]
)