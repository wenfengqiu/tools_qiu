from setuptools import find_packages, setup
import atexit
import shutil


# Function to clean up the 'build' and 'egg-info' folders
def clean_build_folders():
    folders_to_clean = ["build", "tools_qiu.egg-info"]
    for folder in folders_to_clean:
        shutil.rmtree(folder, ignore_errors=True)


# Register the clean_build_folders function with atexit
atexit.register(clean_build_folders)

setup(
    name="tools_qiu",
    version="0.1.1",
    author="Wenfeng Qiu",
    url="https://github.com/wenfengqiu/tools_qiu",
    author_email="wenfengqiu@hotmail.com",
    description="this repo contains a bunch of tools for own use",
    packages=find_packages(),  # Add internal libraries required to be installed
    install_requires=[
        "linearmodels",
        "statsmodels",
    ],  # Add any dependencies your library requires
)
