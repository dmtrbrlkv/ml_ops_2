from catboost import CatBoostClassifier
from streamlit import cache_resource


@cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model('./models/teta_cb.cbm')
    return model