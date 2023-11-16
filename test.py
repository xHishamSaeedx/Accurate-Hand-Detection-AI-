import os, sys
python = sys.executable
print(python)
# get the path of the running script
print(*sys.argv)
os.execl(python, 'test.py', *sys.argv)