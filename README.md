# projekTinter

## How to use docker on your local machine

1. Switch your current directory to directory with Dockerfile
2. Add to dockerfile all scripts you want to run with: COPY {scriptname}.py . (dot at the end too ;) )
3. Run command sudo docker build -t python_docker_base -f Dockerfile . --no-cache
4. Run python console with docker run -it python_docker_base:latest 
5. Or run any script you added in Dockerfile with sudo docker run -it python_docker_base:latest classify-dirt.py

