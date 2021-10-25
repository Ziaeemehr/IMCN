import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='imcn',
    version='0.1',
    author="Abolfazl Ziaeemehr",
    author_email="a.ziaeemehr@gmail.com",
    description="Information measurements on complex networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ziaeemehr/IMCN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    include_package_data=True,
    package_data={'': ['infodynamics.jar']},
)
