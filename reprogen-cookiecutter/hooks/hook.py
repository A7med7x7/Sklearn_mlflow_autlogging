import os
import json

# Path to the cookiecutter context file
context_file = 'cookiecutter.json'

# Get the context from environment variable
project_dir = os.path.realpath(os.path.curdir)
context = {{ cookiecutter | tojson }}

# Set the jupyter_image variable based on gpu_type
if context['gpu_type'] == 'nvidia':
    jupyter_image = 'jupyter-cuda'
elif context['gpu_type'] == 'amd':
    jupyter_image = 'jupyter-rocm'
else:
    jupyter_image = 'jupyter-cpu'

# Write the value to a file in the generated project directory
with open(os.path.join(project_dir, 'jupyter_image.txt'), 'w') as f:
    f.write(jupyter_image)