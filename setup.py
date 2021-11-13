from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup
import os

setup_requires = []

install_requires = []

dependency_links = []


def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#"):
    """Load requirements from a file
    >>> _load_requirements(PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


# PATH_ROOT = "./"
# _load_requirements(PATH_ROOT)
setup(
    name="rlxai",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="srlee",
    author_email="leesungreong@gmail.com",
    description="강화학습에 XAI 적용해보기",
    setup_requires=setup_requires,
    install_requires=install_requires,
    dependency_links=dependency_links,
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    python_requires=">=3.8",
)
