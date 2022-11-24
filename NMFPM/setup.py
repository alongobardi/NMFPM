from setuptools import setup, find_packages


with open ("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name = "nmfpm",
    version = "1.0",
    author = "Alessia Longobardi",
    author_email = "alessia.longobardi@gmail.com",
    description = "NMFPM profile maker",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url="https://github.com/alongobardi/NMFPM",
    package_dir = {"": "."},
    packages={'nmfpm'},
    package_data={"": ['docs/*.json','docs/*.txt']},
    include_package_data=True,
    python_requires=">=3.4",
    install_requires=['numpy','pandas','astropy']

)
