"""Annotation-aware gradient scope for unlabelled HDTF continuation.

MEAD trains every factor. HDTF may train the shared decoder and style/content
parts while explicit affect encoders, condition maps and event prior receive
only MEAD gradients. Shared weights can still change expression behavior, so
this policy must be evaluated; it is not a disentanglement guarantee.
"""
from contextlib import contextmanager


@contextmanager
def content_style_gradient_scope(model):
    modules = (model.teacher, model.student, model.events,
               model.emotion_to_state, model.decoder.affect)
    parameters = {id(p):p for module in modules for p in module.parameters()}
    previous = {key:p.requires_grad for key,p in parameters.items()}
    try:
        for parameter in parameters.values():
            parameter.requires_grad_(False)
        yield
    finally:
        for key,parameter in parameters.items():
            parameter.requires_grad_(previous[key])
