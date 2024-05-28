import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier


def make_pred(dt, path_to_file):
    model = CatBoostClassifier()
    model.load_model('./models/teta_cb.cbm')

    submission = pd.DataFrame({
        'client_id': pd.read_csv(path_to_file)['client_id'],
        'preds': model.predict(dt),
        'proba_1': model.predict_proba(dt)[:, 1]
    })
    return submission


def to_csv(submission):
    csv = submission.drop(columns=['proba_1']).to_csv(index=False).encode('utf-8')
    return csv


def kde(submission):
    fig = plt.figure()
    sns.kdeplot(submission['proba_1'], fill=True, label='Класс 1', color='g')
    plt.title('Плотность распределения предсказанных скоров')
    plt.xlabel('Скор')
    plt.ylabel('Плотность')
    plt.grid()
    plt.legend()
    return fig
