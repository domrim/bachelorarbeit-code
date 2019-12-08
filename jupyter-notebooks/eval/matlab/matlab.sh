#!/bin/sh
set -e

/usr/local/bin/matlab -nodisplay -nosplash -nodesktop -r "run('splitstep.m');exit;" > /dev/null
