from setuptools import find_packages, setup

package_name = 'osracer_policy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/policy.launch.py']),
        ('share/' + package_name + '/config', ['config/policy.yaml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='osrbot',
    maintainer_email='osrbot@example.com',
    description='Sensor-only racing policy node for OSRacer vehicles',
    license='MIT',
    entry_points={'console_scripts': ['policy_node = osracer_policy.policy_node:main']},
)
