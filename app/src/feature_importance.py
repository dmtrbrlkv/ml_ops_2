from catboost import CatBoostClassifier


def get_feature_importance(top=5):
    model = CatBoostClassifier()
    model.load_model('./models/teta_cb.cbm')
    feature_importance = model.get_feature_importance(prettified=True)
    feature_importance = feature_importance[:top]
    return {feature_importance.iloc[i, 0]: feature_importance.iloc[i, 1] for i in range(len(feature_importance))}
