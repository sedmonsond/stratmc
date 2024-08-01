import re
from codecs import open
from os.path import dirname, join, realpath

from setuptools import find_packages, setup

DISTNAME = "stratmc"
DESCRIPTION = "Bayesian statistical framework for reconstructing proxy signals from the stratigraphic record"
AUTHOR = "Stacey Edmonsond"
AUTHOR_EMAIL = "staceyedmonsond777@gmail.com"
URL = "http://github.com/sedmonsond/stratmc"
LICENSE = "GNU General Public License v3.0"



classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.11",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Operating System :: OS Independent",
]

PROJECT_ROOT = dirname(realpath(__file__))

# Get the long description from the README file
with open(join(PROJECT_ROOT, "README.rst"), encoding="utf-8") as buff:
    LONG_DESCRIPTION = buff.read()

REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

test_reqs = ["pytest", "pytest-cov"]

def get_version():
    VERSIONFILE = join("stratmc", "__init__.py")
    lines = open(VERSIONFILE).readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError(f"Unable to find version in {VERSIONFILE}.")


if __name__ == "__main__":
    setup(
        name=DISTNAME,
        version=get_version(),
        maintainer=AUTHOR,
        maintainer_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/x-rst",
        packages=find_packages(),
        include_package_data=False,
        classifiers=classifiers,
        python_requires=">=3.9",
        install_requires=install_reqs,
        tests_require=test_reqs,
    )
