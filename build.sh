homedir="$(pwd)"
echo $homedir

dockerfile="$(pwd)/Dockerfile"

echo $dockerfile

docker build  -t nerfblendshape:devel -f $dockerfile $homedir