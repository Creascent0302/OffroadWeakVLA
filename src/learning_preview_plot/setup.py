from setuptools import setup

package_name = "learning_preview_plot"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="yuheng",
    maintainer_email="yuheng@example.com",
    description="Independent plot node for learning_preview_controller",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "plot_node = learning_preview_plot.plot_node:main",
        ],
    },
)
