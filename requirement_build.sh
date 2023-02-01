#!/bin/bash
if [ $(id -u) != "0" ]; 
then
    echo "You must be the superuser to run this script" >&2
    exit 1
fi
# arch=$(dpkg --print-architecture)
arch=$(arch)
if [ $arch == 'armv7l' ]
then
    echo "Installing dependencies for ARM 32"
    chmod +x ./requirements_arm32.txt
    ./requirements_arm32.txt
elif [ $arch == 'aarch64' ] || [ $arch == 'arm64']
then
    echo "Installing dependecies for ARM 64"
    chmod +x ./requirements_arm64.txt
    ./requirements_arm64.txt
else
    echo "Installing dependencies for Non ARM"
    pip install -r requirements_amd.txt
fi