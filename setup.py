import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requires = []
    for l in f.readlines():
        l = l.strip()
        if "git+" in l:
            url, name = l.split("#egg=")
            install_requires.append(f"{name} @ {url}")
        else:
            install_requires.append(l)

setuptools.setup(
    name="fastfnirs",
    version="0.0.1",
    author="Kalle Makela",
    license="MIT",
    description="A package for fast fNIRS analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    url="https://github.com/Kallemakela/fastfnirs",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
