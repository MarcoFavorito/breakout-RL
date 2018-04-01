# breakout-RL
Experiments on Breakout game applying Reinforcement Learning techniques

## Install
I suggest you to use a virtual environment.
Using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/):

    mkvirtualenv breakout-RL
    
### ALE (deprecated)
Install [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment).

On Ubuntu 16.04, the following steps should work:

    #Clone the official repository
    git clone https://github.com/mgbellemare/Arcade-Learning-Environment.git
    cd Arcade-Learning-Environment
    
    #install dependencies
    sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
    
    #Compile
    mkdir build && cd build
    cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
    make -j 4
    
    #Install the Python module
    pip install .
    
    # return to parent directory
    cd ..
    
### breakout-env
Please install [my fork](https://github.com/MarcoFavorito/breakout-env) of [this repository](https://github.com/SSARCandy/breakout-env):

    pip install git+https://github.com/MarcoFavorito/breakout-env.git@master

### Other requirements

    pip install -r requirements.txt