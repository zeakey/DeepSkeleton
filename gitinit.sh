#!/bin/bash
set -e
rm -rf .git
git init
git add -A
git submodule add https://github.com/zeakey/skeval
git commit -m 'Init'
git remote add origin git@github.com:zeakey/DeepSkeleton
git push -u origin master -f
echo "Done!"

