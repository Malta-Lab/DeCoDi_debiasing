import yaml

# Load the environment.yaml file
with open('environment.yaml', 'r') as f:
    env = yaml.safe_load(f)

# Extract the dependencies
dependencies = env['dependencies']

# Open the requirements.txt file for writing
with open('requirements.txt', 'w') as f:
    for dep in dependencies:
        if isinstance(dep, str):
            f.write(dep + '\n')
        elif isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                f.write(pip_dep + '\n')