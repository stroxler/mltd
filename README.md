# Machine Learning Typing Demo

This repository has some code I used to test-drive the `jaxtyping` library so that
I could verify content for a Pytorch conference poster.

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
