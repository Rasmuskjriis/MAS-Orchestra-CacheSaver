### Setting up WSL copy
1. Go into WSL:
"wsl"

2. Go to Linux Home Directory:
"cd ~"

3. Create a new project folder in Linux:
"mkdir cachesaver"

4. Copy the project from Windows to Linux:
"cp -r "PATH_FOR_PROJECT/." ~/cachesaver"

5. Enter the folder:
"cd ~/cachesaver"

6. Open in VSC:
"code ."

7. Open a terminal in VSC (which should be in WSL), also make sure to open Docker Desktop.

### Setting up Docker
8. Create a Docker image:
"docker build -t cachesaver-base -f orchestrator/Dockerfile.base ."

9. Create a Docker container:
"docker run -it \
  --name cachesaver_dev \
  -v $(pwd):/app \
  --env-file .env \
  cachesaver-base /bin/bash"

(The terminal should now look something like: "root@c73297dabeb0:/app#")

(Use "docker start -ai cachesaver_dev" to reopen container in the future)

10. Setup uv environment:
"uv sync"

### Running the code
11. Create MAS plan via. the orchestrator:
"uv run -m orchestrator.remote_orchestrator"

or run:

"uv pip install --system --no-cache -r pyproject.toml"

and then:

"python3 orchestrator.remote_orchestrator"

12. Execute the orchestrator:
"uv run -m orchestrator.run_mas"

or

"python3 orchestrator.run_mas"