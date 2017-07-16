def _repr_model(model):

    if model._net is None:
        net_representation = '[uninitialised]'
    else:
        net_representation = repr(model._net)

    return ('<{}: {}>'
            .format(
                model.__class__.__name__,
                net_representation,
            ))
