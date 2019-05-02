
from setuptools import find_packages, setup

setup(name="ngs_templates",
      version="0.0.1",
      description="Model templates for NGSolve",
      packages=["ngs_templates"],
      package_dir={"ngs_templates" : "templates"},
      classifiers=("Programming Language :: Python :: 3",
                   "Operating System :: OS Independent",
                   "Development Status :: 2 - Pre-Alpha",
                   "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)")
      )
