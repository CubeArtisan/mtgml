import json

import numpy as np
import hypertools as hyp
from plotly.io import write_html
import plotly.express as px
import pandas as pd

from mtgml.server import get_model

CARD_ADDED_JS = """
const elem = document.getElementById('{plot_id}');
const search = document.createElement('input');
document.body.prepend(search);
search.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        console.log(e.target.value);
        Plotly.restyle(elem, {'marker.size': elem.data.map(({ hovertext }) => hovertext.map((x) => x.includes(e.target.value) ? 32 : 6))}, elem.data.map((_, i) => i));
    }
});
"""

CUBE_ADDED_JS = """
const elem = document.getElementById('{plot_id}');
const search = document.createElement('input');
document.body.prepend(search);
search.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        Plotly.restyle(elem, {'marker.size': elem.data.map(({ customdata }) => customdata.map(([x]) => x.includes(e.target.value) ? 32 : 6))}, elem.data.map((_, i) => i));
    }
});
elem.on('plotly_click', (data) => {
    console.log(data);
    window.location.href = data.points[0].customdata[1];
});
"""

with open('data/maps/int_to_card.json') as fp:
    int_to_card = json.load(fp)

COLOR_MAP = {'w': 'yellow', 'u': 'blue', 'b': 'black', 'r': 'red', 'g': 'green', 'm': 'gold', 'c': 'grey', 'l': 'brown'}

model = get_model()
embeddings = model.embed_cards.embeddings.numpy()
embeddings = embeddings[1:] / np.linalg.norm(embeddings[1:], axis=-1, keepdims=True)
names = [card['name'] for card in int_to_card]
colors = [card['colorcategory'] for card in int_to_card]
reduced = hyp.reduce(embeddings, ndims=2, reduce='UMAP')
df = pd.DataFrame(list(zip(reduced[:, 0], reduced[:, 1], names, colors)), columns=['x', 'y', 'names', 'colors'])
fig = px.scatter(data_frame=df, x='x', y='y', color='colors', hover_name='names', color_discrete_map=COLOR_MAP)
write_html(fig, 'card_embeddings_umap.html', auto_open=True, config={'responsive': True},
           include_plotlyjs='cdn', post_script=CARD_ADDED_JS)
reduced = hyp.reduce(embeddings, ndims=2, reduce={'model': 'TSNE', 'params': {'learning_rate': 'auto', 'init': 'pca'}})
df = pd.DataFrame(list(zip(reduced[:, 0], reduced[:, 1], names, colors)), columns=['x', 'y', 'names', 'colors'])
fig = px.scatter(data_frame=df, x='x', y='y', color='colors', hover_name='names', color_discrete_map=COLOR_MAP)
write_html(fig, 'card_embeddings_tsne.html', auto_open=True, config={'responsive': True},
           include_plotlyjs='cdn', post_script=CARD_ADDED_JS)
reduced = hyp.reduce(embeddings, ndims=2, reduce='PCA')
df = pd.DataFrame(list(zip(reduced[:, 0], reduced[:, 1], names, colors)), columns=['x', 'y', 'names', 'colors'])
fig = px.scatter(data_frame=df, x='x', y='y', color='colors', hover_name='names', color_discrete_map=COLOR_MAP)
write_html(fig, 'card_embeddings_pca.html', auto_open=True, config={'responsive': True},
           include_plotlyjs='cdn', post_script=CARD_ADDED_JS)


df = pd.read_csv('data/cube_data.csv')
with open('data/cube_embeddings.csv') as fp:
    embeddings = np.array([[float(x) for x in line.split(',')] for line in fp.readlines()])
embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
reduced = hyp.reduce(embeddings, ndims=2, reduce={'model': 'TSNE', 'params': {'learning_rate': 'auto', 'init': 'pca'}})
df['x'] = reduced[:, 0]
df['y'] = reduced[:, 1]
fig = px.scatter(data_frame=df, x='x', y='y', hover_name='name', hover_data=('name', 'id'))
write_html(fig, 'cube_embeddings_tsne.html', auto_open=True, config={'responsive': True},
           include_plotlyjs='cdn', post_script=CUBE_ADDED_JS)
reduced = hyp.reduce(embeddings, ndims=2, reduce='UMAP')
df['x'] = reduced[:, 0]
df['y'] = reduced[:, 1]
fig = px.scatter(data_frame=df, x='x', y='y', hover_name='name', hover_data=('name', 'id'))
write_html(fig, 'cube_embeddings_umap.html', auto_open=True, config={'responsive': True},
           include_plotlyjs='cdn', post_script=CUBE_ADDED_JS)
reduced = hyp.reduce(embeddings, ndims=2, reduce='PCA')
df['x'] = reduced[:, 0]
df['y'] = reduced[:, 1]
fig = px.scatter(data_frame=df, x='x', y='y', hover_name='name', hover_data=('name', 'id'))
write_html(fig, 'cube_embeddings_pca.html', auto_open=True, config={'responsive': True},
           include_plotlyjs='cdn', post_script=CUBE_ADDED_JS)
