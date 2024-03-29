"""
This script requires (in addition to volumentations requirements):
    * plotly
    * kaleido
"""
import numpy as np
from volumentations import *
from plotly import graph_objects as go
from plotly import express as px


augmentations = [
    Downscale(.5, .51),
    ElasticTransform((.7, .71)),
    GlassBlur(),
    GridDistortion(distort_limit=.5),
    GridDropout(holes_number_x=3, holes_number_y=3, holes_number_z=3, random_offset=True, fill_value=.5),
    RandomGamma(gamma_limit=(70, 71)),
    RandomScale2(scale_limit=[1.5, 1.6]),
    RotatePseudo2D((1, 2), limit=(40, 41)),
]

X, Y, Z = np.mgrid[-8:8:40j, -8:8:40j, -8:8:40j]
values = np.sin(X*Y*Z) / (X*Y*Z)

fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=.1,
    isomax=.9,
    opacity=.5,
    surface_count=6,
    caps=dict(x_show=False, y_show=False, z_show=False),
    colorscale="gray"
))
fig.write_image("images/original.png")
fig = px.imshow(values[20], color_continuous_scale="gray", zmin=values[20].min(), zmax=values[20].max())
fig.write_image("images/original_flat.png")

for aug in augmentations:
    cube = aug(True, ["image"], image=values)["image"]

    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=cube.flatten(),
        isomin=.1,
        isomax=.9,
        opacity=.5,
        surface_count=6,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale="gray"
    ))
    name = aug.__class__.__name__
    print(f"images/{name}.png")
    fig.write_image(f"images/{name}.png")

    fig = px.imshow(cube[20], color_continuous_scale="gray", zmin=values[20].min(), zmax=values[20].max())
    print(f"images/{name}_flat.png")
    fig.write_image(f"images/{name}_flat.png")
