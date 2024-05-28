import pandas as pd
from src.load_model import load_model
import plotly.figure_factory as ff


def make_pred(dt, path_to_file):
    model = load_model()

    submission = pd.DataFrame({
        'client_id': pd.read_csv(path_to_file)['client_id'],
        'preds': model.predict(dt),
        'proba_1': model.predict_proba(dt)[:, 1]
    })
    return submission


def to_csv(submission):
    csv = submission.drop(columns=['proba_1']).to_csv(index=False).encode('utf-8')
    return csv


def distplot(submission):
    fig = ff.create_distplot([submission['proba_1']], bin_size=0.05, group_labels=['Класс 1'], show_rug=False,
                             colors=['green'])
    fig.update_layout(title_text='Плотность распределения предсказанных скоров')
    return fig
