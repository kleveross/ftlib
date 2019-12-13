from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required_deps = f.read().splitlines()

extras = {}
with open("requirements-dev.txt") as f:
    extras["develop"] = f.read().splitlines()

setup(
    name="ftlib",
    version="master",
    description="A framework to keep data-parallel distributed training continue regardless worker loss or join",
    long_description="FTLib (Fault-Tolerant Library) is a framework to keep data-parallel distributed training "
                     "continue regardless worker loss or join. It exposes collective communication APIs with "
                     "support by gluing a consensus to a communication library, both of which can be user-specific. "
                     "fault-tolerance. A distributed training using FTLib is able to continue as long as at least "
                     "one single worker is alive and when new workers join the training.",
    author=[
        "Wang Zhang",
        "Yuan Tang",
        "Pengcheng Tang",
        "Ce Gao",
        "Deyuan Deng",
    ],
    url="https://github.com/caicloud/ftlib",
    install_requires=required_deps,
    extras_require=extras,
    python_requires=">=3.5",
    packages=find_packages(exclude=["*test*"]),
    package_data={"": ["consensus/shared_storage/proto/communicate.proto"]},
)
