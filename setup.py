from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = "-e ."

def getRequirments(file_path:str)->List[str]:
    """
    this function will retrun the list of requirments
    """
    requirments=[]
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        requirments = [req.replace("\n","") for req in requirments if req != HYPEN_E_DOT]

    return requirments

setup(
    name="MlProject",
    version="0.0.1",
    author="Abdulrhman",
    author_email="abdalrhmanmicro1@gmail.com",
    packages=find_packages(),
    install_requires=getRequirments("requirements.txt")
)