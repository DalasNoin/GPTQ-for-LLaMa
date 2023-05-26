from setuptools import setup

setup(
    name='GPTQ-for-LLaMa',
    version='0.1.0',
    description='A Python package for training and using GPTQ models for LLaMa',
    long_description=open('README.md').read(),
    url='https://github.com/your_username/GPTQ-for-LLaMa',
    author='Your Name',
    author_email='your_email@example.com',
    license='MIT',
    packages=['GPTQ_for_LLaMa'],
    install_requires=[
        safetensors==0.3.0
        datasets==2.10.1
        sentencepiece
        git+https://github.com/huggingface/transformers
        accelerate==0.17.1
        triton==2.0.0
        texttable
        toml
        numpy
        protobuf==3.20.2
    ],
)
