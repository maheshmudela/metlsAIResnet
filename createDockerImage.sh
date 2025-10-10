
#Build the Docker image by running this command:
#docker build -t mahesh1978/repomelts/melts-api .
docker build -t melts-api:1.0 .

#-t custom-tf-api: The tag (-t) gives your image a name, custom-tf-api. The dot (.) tells Docker to look for the Dockerfile in the current directory

#Verify the image by running:
docker images

#ou should see custom-tf-api listed in the output. 


#3. Deploy on your server
#To run your container on a server, you first need to transfer the image and then start the container.
#Transfer the Docker image:
#Option A: Push to a registry. If you have a Docker Hub account, you can push your image and pull it on the server.

# Log in to Docker Hub


docker login -u mahesh1978


# Tag your image with your Docker Hub username
#docker tag mahesh1978/repomelts/melts-api :latest
docker tag melts-api:1.0 mahesh1978/repomelts:1.0

# Push the image to the registry
#docker push mahesh1978/repomelts/melts-api:latest
docker push mahesh1978/repomelts:1.0

#Then, on your server, simply run docker pull your-username/custom-tf-api:latest.

#Option B: Save and load. For a private or local server without a registry, you can save the image to a tarball and transfer it via scp.
#sh
# On your local machine, save the image
#docker save custom-tf-api > custom-tf-api.tar

# Transfer the tarball to your server
#scp custom-tf-api.tar user@your-server-ip:/path/to/project

# On your server, load the image
#docker load < custom-tf-api.tar


#Start the container on your server:
#docker run -d -p 80:5000 --name metls-container melts-api

#docker run -d -p 3000:8080 --name melts-container mahesh1978/repomelts:1.0
docker run -d -p 3000:8080 --name melts-container mahesh1978/repomelts:1.0 python -m uvicorn melts:app --host 0.0.0.0 --port 8080

#-d: Runs the container in detached mode (in the background).
#-p 80:5000: Maps port 80 on your host machine to port 5000 inside the container. This makes your API accessible via HTTP on port 80.
#--name tf-api-container: Gives a friendly name to your container.
#custom-tf-api: The name of the image to run

#Check container status
docker ps


#1. Push your container image to Artifact Registry
#Your GKE cluster needs to access your custom Docker image. It is best practice to use Google's Artifact Registry.

# Tag your local image for Artifact Registry
#docker tag melts-api us-central1-docker.pkg.dev/your-project-id/your-repo/melts-api:latest

# Push the image to Artifact Registry
#docker push us-central1-docker.pkg.dev/your-project-id/your-repo/melts-api:latest




#Using curl (for direct API calls)
#Train the model:
#sh
#curl -X 'POST' 'http://your-server-ip/v1/train'
#Use code with caution.

#Evaluate the model:
#sh
#curl -X 'GET' 'http://your-server-ip/v1/evaluate'
#Use code with caution.

#Predict with an image file:
#sh
#curl -X 'POST' 'http://your-server-ip/v1/predict' \
#-H 'Content-Type: multipart/form-data' \
#-F 'file=@/path/to/your/image.jpg'

#curl -X 'GET' 'http://your-server-ip/v1/get_log' -H 'Accept: application/json' -H 'log-access-key: your-secure-log-key'


#curl -X GET "http://your-server-ip/v1/get_plot" --output my_plot.png

#Replace /path/to/your/image.jpg with the path to an image on your local machine or server.
