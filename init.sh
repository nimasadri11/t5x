rm -rf t5x
git clone https://nimasadri11@github.com/nimasadri11/t5x.git
cd t5x; python3 -m pip install -e '.[tpu]' -f \
    https://storage.googleapis.com/jax-releases/libtpu_releases.html
mkdir -p ~/dir1/user_dir
pip3 install --upgrade jax jaxlib
pip install --upgrade tensorflow-datasets
cp ~/t5x/msmarco_v1/* ~/dir1/user_dir
