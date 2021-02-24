import setuptools
import shutil

VERSION = '0.0.1'

with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

if shutil.which('nvcc') is not None:
    install_requires.append('pycuda')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="libff",
    version=VERSION,
    author="Nathan Pemberton",
    author_email="nathanp@berkeley.edu",
    description="Quick and dirty faas-like experimental system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nathantp/fakefaas.git",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.8',
    include_package_data=True
)
