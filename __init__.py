def classFactory(iface):
    from .core.habitat_classifier import HabitatClassifier
    return HabitatClassifier(iface)