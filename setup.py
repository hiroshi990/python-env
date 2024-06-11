"""building the machine learning 
model as a package so that other can
also use it""" 
from setuptools import find_packages,setup
from typing import List
dot="-e ."
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as fl:
        requirements=fl.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if dot in requirements:
            requirements.remove(dot)
            return requirements        





setup(
    name="mlprojects",
    version="1",
    author="hiroshi",
    author_email="aditya.sharmahdr@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")    
)