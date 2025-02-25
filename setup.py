from setuptools import setup, find_packages

setup(
    name='speechllm',
    version='0.1.0',
    author='Justin Cho',
    author_email='hd.justincho@gmail.com',
    description='Adapting and evaluating LLMs as suitability as backbones for voice assistants',
    long_description='This package facilitates LLM adaptation and evaluation as voice assistants.',
    url='https://github.com/tbd',
    packages=find_packages(),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='voice assistants, language models, evaluation, benchmark',
)