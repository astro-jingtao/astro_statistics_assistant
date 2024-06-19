from typing import Union, List, Callable, Any, Dict

OP_MAP_LABEL: Dict[str, str] = {'log10': r'$\log$', 'square': ''}


class Labels:

    OP_MAP_LABEL = OP_MAP_LABEL

    def __init__(self,
                 labels: Dict[str, str] = None,
                 units: Dict[str, str] = None):
        self.labels: Dict[str, str] = labels if labels is not None else {}
        self.unit_labels: Dict[str, str] = units if units is not None else {}

    def __getitem__(self, key):
        return self.labels[key]

    def __setitem__(self, key, value):
        self.labels[key] = value

    def get(self, name, with_unit=True, op_bracket='{label}') -> str:

        if with_unit:
            if '@' in name:
                _, _name = name.split('@')
            else:
                _name = name
            unit_label = self.unit_labels.get(_name, '')
        else:
            unit_label = ''

        return _get_label_by_name(name,
                                  self.labels,
                                  unit_label,
                                  op_bracket=op_bracket,
                                  op_map_label=self.OP_MAP_LABEL)

    def keys(self):
        return self.labels.keys()

    def values(self):
        return self.labels.values()

    def items(self):
        return self.labels.items()


def _get_label_by_name(name,
                       labels,
                       unit_label,
                       space_before_unit=True,
                       op_bracket='{label}',
                       op_map_label=None) -> str:

    if op_map_label is None:
        op_map_label = OP_MAP_LABEL

    if space_before_unit and (unit_label != ''):
        unit_label = f' {unit_label}'

    if name in labels:
        return labels[name] + unit_label
    if '@' in name:
        op, name = name.split('@')
        label = OP_MAP_LABEL[op] + op_bracket.format(
            label=labels.get(name, name)) + unit_label
    else:
        label = labels.get(name, name) + unit_label
    return label.strip()
