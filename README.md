# Machine Learning Typing Demo

## Setup

Using `uv`:
```
uv init -p 3.13
uv pin 3.13.3  # it seems like uv add starts all over again if you don't do this
uv add torch beartype jaxtyping nptyping docarray pydantic numpydantic ipython jupyter pyrefly
```
To start up jupyter:
```
jupyter lab
```
