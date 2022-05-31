#!/bin/bash

# This is a tool to clean up a Python file and generate reports

filename=$(basename -- "$1")
filename="${filename%.*}"

mkdir checks

# Code formatters
poetry run autoflake --in-place --remove-all-unused-imports --remove-duplicate-keys --remove-unused-variables $1
poetry run black $1
poetry run isort $1

# Security
poetry run bandit --severity-level all --confidence-level all --format txt -o checks/bandit_$filename.txt $1

# Static typing
poetry run mypy $1 > checks/mypy_$filename.txt

# Linting
rm checks/flake8_$filename.txt
poetry run flake8 --output-file checks/flake8_$filename.txt --isolated $1
poetry run pylint --rcfile pylintrc --output checks/pylint_$filename.txt -f text $1
