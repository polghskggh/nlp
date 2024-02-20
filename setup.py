from setuptools import setup

with open('requirements.txt', 'r') as file:
    requirements = [line.strip() for line in file if line.strip()]

setup(
    name='NLP project',
    description='Emotion classification using transformers',
    install_requires=requirements,
)
