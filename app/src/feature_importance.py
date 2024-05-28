from src.load_model import load_model


def get_feature_importance(top=5):
    model = load_model()
    feature_importance = model.get_feature_importance(prettified=True)
    feature_importance = feature_importance[:top]
    return {feature_importance.iloc[i, 0]: feature_importance.iloc[i, 1] for i in range(len(feature_importance))}
