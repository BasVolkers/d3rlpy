#! /bin/bash

# zip -rq all.zip * -x assets/\* -x .github/\* tests/\* reproductions/\* tutorials/\* docker/\* examples/\* scripts/\*
# Make mechanical ventilation folder
# ssh delftblue "mkdir -p /home/bvolkers/d3rlpy && exit"

# Transfer files to delftblue and unzip
# scp all.zip  delftblue:/home/bvolkers/d3rlpy/all.zip
# ssh delftblue "cd /home/bvolkers/d3rlpy && unzip -q -o all.zip && exit"

scp -r d3rlpy delftblue:/home/bvolkers/d3rlpy

# Remove zip file
rm all.zip