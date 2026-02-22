#!/bin/sh

biber main
xelatex -interaction=nonstopmode main.tex
