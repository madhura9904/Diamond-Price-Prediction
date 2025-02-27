from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:#Read all packages in requirements
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]


    return requirements



setup(

    name='DiamondPricePrediction',
    version='0.0.1',
    author='pwskills',
    author_email='pw@gmailcom',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)

from setuptools import find_packages, setup

setup(
    name="diamond_price_prediction",
    version="0.1",
    packages=find_packages()
)